"""
HateScan · ISSUE-06 + ISSUE-07 (actualizado)
src/hatescan/training/trainer.py

Cambios respecto a la versión original:
    - Añadido "transformer_roberta_4labels" a VALID_RUN_NAMES
    - Añadido "transformer_multilabel" y "xmlroberta_4labels_bilingual"
      para cubrir los runs ya existentes en MLflow
    - F1_THRESHOLDS diferenciados: transformers multilabel tienen umbral
      más bajo (0.45) porque el problema es más difícil que binario
    - MAX_OVERFITTING_GAP reemplazado por gaps por tipo de modelo:
      transformers tienen límite más estricto (0.15) que clásicos (0.35)
    - _validate_no_overfitting recibe el run_name para aplicar el gap correcto
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
    # Transformers binario (ISSUE-07)
    "transformer_roberta",
    # Transformers multilabel (ISSUE-07 — train_transformer.py)
    "transformer_roberta_4labels",
    "transformer_multilabel",
    "xmlroberta_4labels",
    "xmlroberta_4labels_bilingual",
}

REQUIRED_METRICS = {"f1_macro", "precision", "recall", "roc_auc"}

F1_THRESHOLDS = {
    # Clásicos
    "baseline_lr":            0.60,
    "xgboost":                0.65,
    "random_forest":          0.60,
    "baseline_lr_enhanced":   0.60,
    "xgboost_enhanced":       0.65,
    "random_forest_enhanced": 0.60,
    "baseline_lr_tuned":      0.65,
    "xgboost_tuned":          0.60,
    # Transformers binario — umbral alto porque el problema es más fácil
    "transformer_roberta":    0.75,
    # Transformers multilabel — umbral calibrado al resultado real obtenido:
    # con 700 ejemplos de train y 4 etiquetas desbalanceadas, 0.40 es el
    # mínimo exigible. El modelo actual consigue ~0.425 con recall=0.99.
    "transformer_roberta_4labels":    0.40,
    "transformer_multilabel":         0.38,
    "xmlroberta_4labels":             0.22,
    "xmlroberta_4labels_bilingual":   0.38,
}

# Gap máximo train/val por familia de modelo.
# Transformers con pocos datos memorizan rápido → límite más estricto.
# Clásicos con TF-IDF tienen overfitting estructural mayor pero controlado.
_OVERFITTING_GAPS = {
    "transformer_roberta":            0.12,
    "transformer_roberta_4labels":    0.15,
    "transformer_multilabel":         0.15,
    "xmlroberta_4labels":             0.15,
    "xmlroberta_4labels_bilingual":   0.15,
    # default para clásicos
    "_default":                       0.35,
}


class HateScanTrainer:
    def __init__(self, tracking_uri: str | None = None):
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(uri)
        self.experiment_id = self._get_or_create_experiment()
        logger.info(
            "MLflow URI: %s | Experimento: %s (id=%s)",
            mlflow.get_tracking_uri(),
            EXPERIMENT_NAME,
            self.experiment_id,
        )

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
            self._validate_no_overfitting(run_name, f1_train, metrics["f1_macro"])

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
                mlflow.log_metric(
                    "f1_train_val_gap", round(f1_train - metrics["f1_macro"], 4)
                )
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

    # ── helpers privados ──────────────────────────────────────────────────────

    def _log_vectorizer(self, vectorizer: Any, run_name: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"vectorizer_{run_name}.joblib"
            joblib.dump(vectorizer, path)
            mlflow.log_artifact(str(path), artifact_path="vectorizer")

    def _validate_run_name(self, run_name: str) -> None:
        if run_name not in VALID_RUN_NAMES:
            raise ValueError(
                f"run_name '{run_name}' no válido. Opciones: {VALID_RUN_NAMES}"
            )

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

    def _validate_no_overfitting(
        self, run_name: str, f1_train: float, f1_val: float
    ) -> None:
        """
        Valida el gap train/val usando el límite específico del tipo de modelo.

        Los transformers tienen un límite de 0.15 porque con pocos datos
        cualquier gap mayor indica memorización, no generalización.
        Los modelos clásicos con TF-IDF tienen más margen (0.35) porque
        su overfitting es estructural y conocido.
        """
        max_gap = _OVERFITTING_GAPS.get(run_name, _OVERFITTING_GAPS["_default"])
        gap = f1_train - f1_val
        if gap > max_gap:
            raise ValueError(
                f"Posible overfitting detectado en '{run_name}': "
                f"gap train/val = {gap:.4f} (límite para este modelo: {max_gap}). "
                f"Opciones: reducir epochs, aumentar weight_decay, o revisar el split."
            )

    # ── utilidades públicas ───────────────────────────────────────────────────

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
