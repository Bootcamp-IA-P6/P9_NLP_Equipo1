"""
HateScan · ISSUE-06
src/hatescan/training/register_models.py
 
Registra en MLflow los modelos de los tres enfoques de ISSUE-05:
  - baseline.py              → baseline_lr, xgboost, random_forest
  - enhanced.py              → baseline_lr_enhanced, xgboost_enhanced, random_forest_enhanced
  - hyperparameter_tuning.py → baseline_lr_tuned, xgboost_tuned
    (requiere que haya ejecutado hyperparameter_tuning.py primero)
"""
import logging
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
 
from src.hatescan.training.trainer import HateScanTrainer
 
logger = logging.getLogger(__name__)
 
TUNING_ARTIFACT_DIR = Path("models/artifacts/baseline_tuned")
 
 
def _compute_metrics(model, X, y) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1_macro":  round(f1_score(y, y_pred, average="macro"), 4),
        "precision": round(precision_score(y, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, average="macro", zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_proba), 4),
    }
 
 
def _register_model(trainer, run_name, model, vectorizer, X_train, X_val, X_test, y_train, y_val, y_test):
    """Helper: calcula métricas y registra un modelo en MLflow."""
    val_metrics   = _compute_metrics(model, X_val,   y_val)
    train_metrics = _compute_metrics(model, X_train, y_train)
    test_metrics  = _compute_metrics(model, X_test,  y_test)
 
    all_metrics = {
        **val_metrics,
        "f1_macro_test":  test_metrics["f1_macro"],
        "precision_test": test_metrics["precision"],
        "recall_test":    test_metrics["recall"],
        "roc_auc_test":   test_metrics["roc_auc"],
    }
 
    run_id = trainer.log_run(
        run_name=run_name,
        model=model,
        vectorizer=vectorizer,
        params=model.get_params(),
        metrics=all_metrics,
        X_sample=X_val,
        y_sample=y_val,
        f1_train=train_metrics["f1_macro"],
    )
 
    gap = train_metrics["f1_macro"] - val_metrics["f1_macro"]
    logger.info(
        "%s → run_id=%s | F1_val=%.4f | F1_test=%.4f | gap=%.4f %s",
        run_name, run_id,
        val_metrics["f1_macro"], test_metrics["f1_macro"], gap,
        "⚠️ OVERFITTING" if gap > 0.05 else "✅",
    )
 
 
def _get_splits(df):
    """Splits 70/15/15 + vectorizador a partir de un DataFrame preprocesado."""
    from src.hatescan.training.baseline import (
        split_dataset, build_vectorizer, transform_corpus
    )
    train_df, val_df, test_df = split_dataset(df)
    vectorizer, X_train = build_vectorizer(train_df["text_processed"])
    X_val  = transform_corpus(vectorizer, val_df["text_processed"])
    X_test = transform_corpus(vectorizer, test_df["text_processed"])
    y_train = train_df["IsToxic"]
    y_val   = val_df["IsToxic"]
    y_test  = test_df["IsToxic"]
    return vectorizer, X_train, X_val, X_test, y_train, y_val, y_test
 
 
def register_from_baseline(
    include_random_forest: bool = True,
    trainer_instance: HateScanTrainer | None = None,
):
    """Registra los modelos de baseline.py."""
    from src.hatescan.training.baseline import (
        load_dataset, preprocess_text,
        train_logistic_regression, train_xgboost,
        train_random_forest, compute_class_weights,
    )
    trainer = trainer_instance or HateScanTrainer()
    logger.info("=== Registrando baseline ===")
 
    df = preprocess_text(load_dataset())
    vectorizer, X_train, X_val, X_test, y_train, y_val, y_test = _get_splits(df)
    spw = compute_class_weights(y_train)
 
    _register_model(trainer, "baseline_lr",
        train_logistic_regression(X_train, y_train),
        vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)
 
    _register_model(trainer, "xgboost",
        train_xgboost(X_train, y_train, spw),
        vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)
 
    if include_random_forest:
        _register_model(trainer, "random_forest",
            train_random_forest(X_train, y_train),
            vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)
 
 
def register_from_enhanced(
    include_random_forest: bool = True,
    trainer_instance: HateScanTrainer | None = None,
):
    """Registra los modelos de enhanced.py (con features extra)."""
    from src.hatescan.training.enhanced import (
        load_dataset, preprocess_text, extract_text_features,
        split_dataset, build_vectorizer, transform_corpus,
        train_logistic_regression, train_xgboost,
        train_random_forest, compute_class_weights,
    )
    from scipy.sparse import hstack
    from sklearn.preprocessing import StandardScaler
 
    trainer = trainer_instance or HateScanTrainer()
    logger.info("=== Registrando enhanced ===")
 
    df = extract_text_features(preprocess_text(load_dataset()))
    train_df, val_df, test_df = split_dataset(df)
 
    vectorizer, X_train_tfidf = build_vectorizer(train_df["text_processed"])
    X_val_tfidf  = transform_corpus(vectorizer, val_df["text_processed"])
    X_test_tfidf = transform_corpus(vectorizer, test_df["text_processed"])
 
    extra_cols = ["text_length", "word_count", "has_uppercase",
                  "all_caps_ratio", "exclamation_count", "question_count"]
    scaler = StandardScaler()
    X_train_extra = scaler.fit_transform(train_df[extra_cols].values)
    X_val_extra   = scaler.transform(val_df[extra_cols].values)
    X_test_extra  = scaler.transform(test_df[extra_cols].values)
 
    X_train = hstack([X_train_tfidf, X_train_extra])
    X_val   = hstack([X_val_tfidf,   X_val_extra])
    X_test  = hstack([X_test_tfidf,  X_test_extra])
 
    y_train = train_df["IsToxic"]
    y_val   = val_df["IsToxic"]
    y_test  = test_df["IsToxic"]
    spw = compute_class_weights(y_train)
 
    _register_model(trainer, "baseline_lr_enhanced",
        train_logistic_regression(X_train, y_train),
        vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)
 
    _register_model(trainer, "xgboost_enhanced",
        train_xgboost(X_train, y_train, spw),
        vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)
 
    if include_random_forest:
        _register_model(trainer, "random_forest_enhanced",
            train_random_forest(X_train, y_train),
            vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)
 
 
def register_from_tuned(trainer_instance=None):
    """
    Registra los modelos optimizados de hyperparameter_tuning.py.
    REQUIERE que tu compañero haya ejecutado hyperparameter_tuning.py primero
    para que existan los archivos en models/artifacts/baseline_tuned/.
    """
    trainer = trainer_instance or HateScanTrainer()
    logger.info("=== Registrando tuned ===")

    lr_path  = TUNING_ARTIFACT_DIR / "logistic_regression.joblib"
    xgb_path = TUNING_ARTIFACT_DIR / "xgboost_classifier.joblib"
    vec_path = TUNING_ARTIFACT_DIR / "tfidf_vectorizer.joblib"

    for path in [lr_path, xgb_path, vec_path]:
        if not path.exists():
            raise FileNotFoundError(f"No encontrado: {path}")

    lr_tuned   = joblib.load(lr_path)
    xgb_tuned  = joblib.load(xgb_path)
    vectorizer = joblib.load(vec_path)  # ← vectorizador del tuning, no el estándar

    # Splits usando el vectorizador tuned
    from src.hatescan.training.baseline import load_dataset, preprocess_text, split_dataset
    df = preprocess_text(load_dataset())
    train_df, val_df, test_df = split_dataset(df)

    # fit NO — solo transform, el vectorizador ya está entrenado
    X_train = vectorizer.transform(train_df["text_processed"])
    X_val   = vectorizer.transform(val_df["text_processed"])
    X_test  = vectorizer.transform(test_df["text_processed"])

    y_train = train_df["IsToxic"]
    y_val   = val_df["IsToxic"]
    y_test  = test_df["IsToxic"]

    _register_model(trainer, "baseline_lr_tuned",
        lr_tuned, vectorizer,
        X_train, X_val, X_test, y_train, y_val, y_test)

    _register_model(trainer, "xgboost_tuned",
        xgb_tuned, vectorizer,
        X_train, X_val, X_test, y_train, y_val, y_test)
 
 
def register_all(
    include_random_forest: bool = True,
    include_tuned: bool = False,
    trainer_instance: HateScanTrainer | None = None,
):
    """
    Registra todos los modelos en MLflow.
 
    Args:
        include_random_forest: incluir RF en baseline y enhanced.
        include_tuned:         incluir modelos optimizados con Optuna.
                               Solo activar si tu compañero ya ejecutó
                               hyperparameter_tuning.py.
    """
    trainer = trainer_instance or HateScanTrainer()
 
    register_from_baseline(include_random_forest=include_random_forest, trainer_instance=trainer)
    register_from_enhanced(include_random_forest=include_random_forest, trainer_instance=trainer)
 
    if include_tuned:
        register_from_tuned(trainer_instance=trainer)
 
    logger.info("\n════════════════════════════════")
    logger.info("RESUMEN FINAL (ordenado por F1_val)")
    logger.info("════════════════════════════════")
    for run in trainer.list_runs():
        logger.info("%-30s F1_val=%.4f", run["run_name"], run["f1_macro"] or 0)
 
    best = trainer.get_best_run()
    if best:
        logger.info("\n🏆 Mejor modelo: %s (F1_val=%.4f)",
            best["run_name"], best["metrics"]["f1_macro"])