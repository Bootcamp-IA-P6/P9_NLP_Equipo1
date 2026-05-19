"""
HateScan · ISSUE-06
src/hatescan/training/trainer.py
"""
 
import os
import logging
import tempfile
from pathlib import Path
from typing import Any
 
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
 
logger = logging.getLogger(__name__)
 
EXPERIMENT_NAME = "hatescan_nlp"
 
VALID_RUN_NAMES = {
    # baseline.py
    "baseline_lr",
    "xgboost",
    "random_forest",
    # enhanced.py
    "baseline_lr_enhanced",
    "xgboost_enhanced",
    "random_forest_enhanced",
    # hyperparameter_tuning.py
    "baseline_lr_tuned",
    "xgboost_tuned",
    # ISSUE-07
    "transformer_roberta",
}
 
REQUIRED_METRICS = {"f1_macro", "precision", "recall", "roc_auc"}
 
F1_THRESHOLDS = {
    "baseline_lr":            0.60,
    "xgboost":                0.65,
    "random_forest":          0.60,
    "baseline_lr_enhanced":   0.60,
    "xgboost_enhanced":       0.65,
    "random_forest_enhanced": 0.60,
    "baseline_lr_tuned":      0.65,
    "xgboost_tuned":          0.60,
    "transformer_roberta":    0.75,
}
 
MAX_OVERFITTING_GAP = 0.35
 
 
class HateScanTrainer:
    def __init__(self, tracking_uri: str | None = None):
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(uri)
        self.experiment_id = self._get_or_create_experiment()
        logger.info("MLflow URI: %s | Experimento: %s (id=%s)",
                    mlflow.get_tracking_uri(), EXPERIMENT_NAME, self.experiment_id)
 
    def _get_or_create_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                tags={
                    "project": "HateScan",
                    "task": "hate_speech_classification",
                    "metric_principal": "f1_macro",
                },
            )
            logger.info("Experimento creado: %s", EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
            logger.info("Experimento existente: %s", EXPERIMENT_NAME)
        return experiment_id
 
    def log_run(
        self,
        run_name: str,
        model: Any,
        vectorizer: Any | None,
        params: dict[str, Any],
        metrics: dict[str, float],
        X_sample: Any | None = None,
        y_sample: Any | None = None,
        train_duration_seconds: float | None = None,
        f1_train: float | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        self._validate_run_name(run_name)
        self._validate_metrics(metrics)
        self._validate_f1_threshold(run_name, metrics["f1_macro"])
        if f1_train is not None:
            self._validate_no_overfitting(f1_train, metrics["f1_macro"])
 
        all_tags = {"run_type": run_name, **(tags or {})}
 
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=all_tags,
        ) as run:
            mlflow.log_params(params)
            if vectorizer is not None:
                mlflow.log_param("vectorizer_type", type(vectorizer).__name__)
 
            mlflow.log_metrics(metrics)
 
            if f1_train is not None:
                mlflow.log_metric("f1_macro_train", f1_train)
                mlflow.log_metric("f1_train_val_gap", round(f1_train - metrics["f1_macro"], 4))
            if train_duration_seconds is not None:
                mlflow.log_metric("train_duration_seconds", train_duration_seconds)
 
            signature = None
            if X_sample is not None and y_sample is not None:
                try:
                    preds = model.predict(X_sample)
                    signature = infer_signature(X_sample, preds)
                except Exception as e:
                    logger.warning("No se pudo inferir signature: %s", e)
 
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"hatescan_{run_name}",
            )
 
            if vectorizer is not None:
                self._log_vectorizer(vectorizer, run_name)
 
            run_id = run.info.run_id
            logger.info("Run registrado: %s (run_id=%s)", run_name, run_id)
            return run_id
 
    def _log_vectorizer(self, vectorizer: Any, run_name: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"vectorizer_{run_name}.joblib"
            joblib.dump(vectorizer, path)
            mlflow.log_artifact(str(path), artifact_path="vectorizer")
 
    def _validate_run_name(self, run_name: str) -> None:
        if run_name not in VALID_RUN_NAMES:
            raise ValueError(f"run_name '{run_name}' no válido. Opciones: {VALID_RUN_NAMES}")
 
    def _validate_metrics(self, metrics: dict[str, float]) -> None:
        missing = REQUIRED_METRICS - set(metrics.keys())
        if missing:
            raise ValueError(f"Faltan métricas obligatorias: {missing}")
 
    def _validate_f1_threshold(self, run_name: str, f1_macro: float) -> None:
        threshold = F1_THRESHOLDS.get(run_name, 0.0)
        if f1_macro < threshold:
            raise ValueError(
                f"F1-macro {f1_macro:.4f} no supera el umbral mínimo "
                f"{threshold} para '{run_name}'."
            )
 
    def _validate_no_overfitting(self, f1_train: float, f1_val: float) -> None:
        gap = f1_train - f1_val
        if gap > MAX_OVERFITTING_GAP:
            raise ValueError(
                f"Posible overfitting: gap train/val = {gap:.4f} (> {MAX_OVERFITTING_GAP}). "
                "Revisa regularización o el pipeline de preprocesamiento."
            )
 
    def list_runs(self) -> list[dict]:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["metrics.f1_macro DESC"],
        )
        return [
            {
                "run_id":    r.info.run_id,
                "run_name":  r.data.tags.get("run_type", "unknown"),
                "status":    r.info.status,
                "f1_macro":  r.data.metrics.get("f1_macro"),
                "precision": r.data.metrics.get("precision"),
                "recall":    r.data.metrics.get("recall"),
                "roc_auc":   r.data.metrics.get("roc_auc"),
            }
            for r in runs
        ]
 
    def get_best_run(self, metric: str = "f1_macro") -> dict | None:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None
        best = runs[0]
        return {
            "run_id":   best.info.run_id,
            "run_name": best.data.tags.get("run_type", "unknown"),
            "metrics":  best.data.metrics,
            "params":   best.data.params,
        }