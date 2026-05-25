"""
HateScan · ISSUE-07
src/hatescan/training/register_all_transformers.py

Registra todos los modelos Transformer en MLflow.
Funciona para modelos binarios y multilabel.

Ejecutar:
    python -m src.hatescan.training.register_all_transformers
"""

import json
import logging
from pathlib import Path
import mlflow
from src.hatescan.training.trainer import HateScanTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración de modelos ──────────────────────────────────────────────────
# Añade o quita modelos de esta lista según lo que tengas en models/artifacts/
TRANSFORMER_MODELS = [
    {
        "run_name":  "transformer_roberta",
        "path":      "models/artifacts/transformer",
        "model_type": "binary",
    },
    {
        "run_name":  "transformer_multilabel",
        "path":      "models/artifacts/transformer_multilabel",
        "model_type": "multilabel",
    },
    {
        "run_name":  "transformer_roberta_4labels",
        "path":      "models/artifacts/transformer_roberta_4labels",
        "model_type": "multilabel",
    },
    {
        "run_name":  "xmlroberta_4labels",
        "path":      "models/artifacts/xmlroberta_4labels",
        "model_type": "multilabel",
    },
    {
        "run_name":  "xmlroberta_4labels_bilingual",
        "path":      "models/artifacts/xmlroberta_4labels_bilingual",
        "model_type": "multilabel",
    },
]


def build_mlflow_metrics(metrics: dict, model_type: str) -> dict:
    """
    Construye el dict de métricas para MLflow a partir del metrics.json de Colab.
    Funciona para modelos binarios y multilabel.
    """
    mlflow_metrics = {}

    # ── Métricas val ──────────────────────────────────────────────────────────
    mlflow_metrics["f1_macro"]   = metrics.get("f1_macro_val",   0.0)
    mlflow_metrics["precision"]  = metrics.get("precision_val",  0.0)
    mlflow_metrics["recall"]     = metrics.get("recall_val",     0.0)

    # roc_auc solo en binario
    if model_type == "binary" and "roc_auc_val" in metrics:
        mlflow_metrics["roc_auc"] = metrics["roc_auc_val"]

    # ── Métricas test ─────────────────────────────────────────────────────────
    mlflow_metrics["f1_macro_test"]  = metrics.get("f1_macro_test",  0.0)
    mlflow_metrics["precision_test"] = metrics.get("precision_test", 0.0)
    mlflow_metrics["recall_test"]    = metrics.get("recall_test",    0.0)

    if model_type == "binary" and "roc_auc_test" in metrics:
        mlflow_metrics["roc_auc_test"] = metrics["roc_auc_test"]

    if model_type == "multilabel":
        mlflow_metrics["f1_micro_test"]   = metrics.get("f1_micro_test",   0.0)
        mlflow_metrics["f1_samples_test"] = metrics.get("f1_samples_test", 0.0)

    # ── Overfitting ───────────────────────────────────────────────────────────
    mlflow_metrics["f1_macro_train"]   = metrics.get("f1_macro_train",   0.0)
    mlflow_metrics["f1_train_val_gap"] = metrics.get("f1_train_val_gap", 0.0)

    # ── Latencia ──────────────────────────────────────────────────────────────
    latency_key = "mean_latency_cpu_seconds" if "mean_latency_cpu_seconds" in metrics \
                  else "latency_cpu_seconds"
    mlflow_metrics["latency_cpu_seconds"] = metrics.get(latency_key, 0.0)

    return mlflow_metrics


def build_mlflow_params(metrics: dict, model_type: str) -> dict:
    """Construye los parámetros a loguear en MLflow."""
    params = {
        "model_type":  model_type,
        "model_name":  metrics.get("model_name", "unknown"),
    }
    if "labels" in metrics:
        params["labels"]     = str(metrics["labels"])
        params["num_labels"] = len(metrics["labels"])
    if "thresholds" in metrics:
        params["thresholds"] = str(metrics["thresholds"])
    return params


def register_transformer(model_config: dict, trainer: HateScanTrainer) -> str:
    """Registra un modelo Transformer en MLflow."""
    run_name   = model_config["run_name"]
    model_path = Path(model_config["path"])
    model_type = model_config["model_type"]

    # Validar que el modelo existe
    if not model_path.exists():
        logger.warning("⚠️ No encontrado: %s — saltando", model_path)
        return None

    metrics_path = model_path / "metrics.json"
    if not metrics_path.exists():
        logger.warning("⚠️ metrics.json no encontrado en %s — saltando", model_path)
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    logger.info("Registrando: %s | F1_val=%.4f | F1_test=%.4f | gap=%.4f",
                run_name,
                metrics.get("f1_macro_val", 0),
                metrics.get("f1_macro_test", 0),
                metrics.get("f1_train_val_gap", 0))

    mlflow_metrics = build_mlflow_metrics(metrics, model_type)
    mlflow_params  = build_mlflow_params(metrics, model_type)

    gap = metrics.get("f1_train_val_gap", 0)

    with mlflow.start_run(
        experiment_id=trainer.experiment_id,
        run_name=run_name,
        tags={
            "run_type":   run_name,
            "model_type": model_type,
            "overfitting": "⚠️" if gap > 0.05 else "✅",
        },
    ) as run:
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)

        # Artefactos
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        if (model_path / "per_label_metrics").exists() or "per_label_metrics" in metrics:
            per_label_path = model_path / "metrics.json"
            mlflow.log_artifact(str(per_label_path), artifact_path="per_label_metrics")
        mlflow.log_artifacts(str(model_path), artifact_path="model")

        run_id = run.info.run_id
        logger.info("%s → run_id=%s | gap=%.4f %s",
                    run_name, run_id, gap,
                    "✅" if gap <= 0.05 else "⚠️ OVERFITTING")
        return run_id


def register_all_transformers():
    """Registra todos los modelos Transformer configurados arriba."""
    trainer = HateScanTrainer()

    logger.info("=== Registrando Transformers ===")
    results = []
    for config in TRANSFORMER_MODELS:
        run_id = register_transformer(config, trainer)
        if run_id:
            results.append({"run_name": config["run_name"], "run_id": run_id})

    # Resumen
    logger.info("\n════════════════════════════════════════")
    logger.info("RESUMEN TRANSFORMERS (por F1_val)")
    logger.info("════════════════════════════════════════")
    runs = trainer.list_runs()
    transformer_runs = [r for r in runs if any(
        t in r["run_name"] for t in
        ["transformer", "roberta", "xmlroberta"]
    )]
    for run in transformer_runs:
        logger.info("%-35s F1_val=%.4f", run["run_name"], run["f1_macro"] or 0)

    best = trainer.get_best_run()
    if best:
        logger.info("\n🏆 Mejor modelo overall: %s (F1_val=%.4f)",
                    best["run_name"], best["metrics"]["f1_macro"])


if __name__ == "__main__":
    register_all_transformers()
