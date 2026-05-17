"""
HateScan · ISSUE-06
src/hatescan/training/register_models.py
"""
import logging
from src.hatescan.training.trainer import HateScanTrainer
from src.hatescan.training.nlp_models import compute_metrics

logger = logging.getLogger(__name__)


def register_all_models(
    baseline_lr, xgboost_model, rf_model, tfidf_vectorizer,
    X_train, X_val, X_test, y_train, y_val, y_test,
    results: dict | None = None,
    trainer_instance: HateScanTrainer | None = None,
    **kwargs,
):
    """
    Registra los tres modelos en MLflow.
    - Métricas sobre VAL SET  → para comparar modelos.
    - F1 sobre TRAIN          → para validar overfitting.
    - Métricas sobre TEST     → sufijo _test, para informe final.
    """
    trainer = trainer_instance or HateScanTrainer()

    models = [
        ("baseline_lr",   baseline_lr),
        ("xgboost",       xgboost_model),
        ("random_forest", rf_model),
    ]

    for run_name, model in models:
        logger.info("Registrando: %s", run_name)

        val_metrics   = compute_metrics(model, X_val,   y_val)
        train_metrics = compute_metrics(model, X_train, y_train)
        test_metrics  = compute_metrics(model, X_test,  y_test)

        all_metrics = {
            **val_metrics,
            "f1_macro_test":  test_metrics["f1_macro"],
            "precision_test": test_metrics["precision"],
            "recall_test":    test_metrics["recall"],
            "roc_auc_test":   test_metrics["roc_auc"],
        }

        train_secs = results.get(run_name, {}).get("train_secs") if results else None
        if results and run_name == "xgboost":
            train_secs = results.get("xgboost", {}).get("train_secs")

        run_id = trainer.log_run(
            run_name=run_name,
            model=model,
            vectorizer=tfidf_vectorizer,
            params=model.get_params(),
            metrics=all_metrics,
            X_sample=X_val,
            y_sample=y_val,
            train_duration_seconds=train_secs,
            f1_train=train_metrics["f1_macro"],
        )

        gap = train_metrics["f1_macro"] - val_metrics["f1_macro"]
        logger.info(
            "%s → run_id=%s | F1_val=%.4f | F1_test=%.4f | gap=%.4f %s",
            run_name, run_id,
            val_metrics["f1_macro"], test_metrics["f1_macro"], gap,
            "⚠️ OVERFITTING" if gap > 0.05 else "✅",
        )

    logger.info("\n── Resumen (por F1_val) ──")
    for run in trainer.list_runs():
        logger.info("%-20s F1_val=%.4f  P=%.4f  R=%.4f",
            run["run_name"], run["f1_macro"] or 0,
            run["precision"] or 0, run["recall"] or 0)

    best = trainer.get_best_run()
    if best:
        logger.info("Mejor modelo: %s (F1_val=%.4f)", best["run_name"], best["metrics"]["f1_macro"])
