"""
HateScan · ISSUE-06
src/hatescan/training/trainer.py

Wrapper de MLflow para HateScan. Crea/reutiliza el experimento
'hatescan_nlp' y registra parámetros, métricas y artefactos por run.

Uso:
    from src.hatescan.training.trainer import HateScanTrainer

    trainer = HateScanTrainer()
    run_id = trainer.log_run(
        run_name="xgboost",
        model=xgboost_model,
        vectorizer=tfidf_vectorizer,
        params=xgboost_model.get_params(),
        metrics={"f1_macro": 0.77, "precision": 0.75, "recall": 0.79, "roc_auc": 0.88},
        f1_train=0.80,
        train_duration_seconds=12.4,
    )
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
    "baseline_lr",
    "xgboost",
    "random_forest",
    "transformer_roberta",  # reservado para ISSUE-07
}

REQUIRED_METRICS = {"f1_macro", "precision", "recall", "roc_auc"}

# Umbrales mínimos de F1-macro sobre VAL SET por tipo de modelo.
# Nota: el plan define los umbrales sobre test (LR>0.70, XGB>0.75, Transformer>0.80).
# Sobre val set se permite un margen de -0.05 para no bloquear runs válidos.
F1_THRESHOLDS = {
    "baseline_lr":        0.65,
    "xgboost":            0.70,
    "random_forest":      0.65,
    "transformer_roberta": 0.75,
}

# Gap máximo F1 train/val antes de considerar overfitting
MAX_OVERFITTING_GAP = 0.05


class HateScanTrainer:
    """
    Wrapper de MLflow para HateScan.

    - Crea / reutiliza el experimento 'hatescan_nlp'.
    - Loguea parámetros, métricas (val + test) y artefactos en cada run.
    - Registra el modelo en el MLflow Model Registry.
    - Valida run_name, métricas obligatorias, umbral F1 y gap overfitting.
    """

    def __init__(self, tracking_uri: str | None = None):
        """
        Args:
            tracking_uri: URI del servidor MLflow.
                - None  → lee MLFLOW_TRACKING_URI del entorno,
                          o usa 'sqlite:///mlruns.db' local por defecto.
                - str   → ruta local o servidor remoto ('http://...').
        """
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(uri)
        self.experiment_id = self._get_or_create_experiment()
        logger.info("MLflow URI: %s | Experimento: %s (id=%s)",
                    mlflow.get_tracking_uri(), EXPERIMENT_NAME, self.experiment_id)

    # ── Experimento ────────────────────────────────────────────────────────────

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

    # ── Log principal ──────────────────────────────────────────────────────────

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
        """
        Registra un run completo en MLflow.

        Args:
            run_name:               Identificador del run. Debe estar en VALID_RUN_NAMES.
            model:                  Modelo sklearn entrenado.
            vectorizer:             TF-IDF u otro vectorizador (None para Transformer).
            params:                 Hiperparámetros del modelo (model.get_params()).
            metrics:                Métricas sobre VAL SET: f1_macro, precision, recall, roc_auc.
                                    Opcionalmente también métricas _test para informe final.
            X_sample:               Muestra de X para inferir la firma del modelo (opcional).
            y_sample:               Muestra de y para inferir la firma del modelo (opcional).
            train_duration_seconds: Tiempo de entrenamiento en segundos (opcional).
            f1_train:               F1-macro sobre train — para validar overfitting (opcional).
            tags:                   Tags MLflow adicionales (opcional).

        Returns:
            run_id (str)

        Raises:
            ValueError: run_name inválido, métricas faltantes,
                        F1 bajo umbral mínimo, o gap overfitting > 5 pp.
        """
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
            # Parámetros
            mlflow.log_params(params)
            if vectorizer is not None:
                mlflow.log_param("vectorizer_type", type(vectorizer).__name__)

            # Métricas (val + test si vienen)
            mlflow.log_metrics(metrics)

            # Métricas extra de train / overfitting
            if f1_train is not None:
                mlflow.log_metric("f1_macro_train", f1_train)
                mlflow.log_metric(
                    "f1_train_val_gap",
                    round(f1_train - metrics["f1_macro"], 4),
                )
            if train_duration_seconds is not None:
                mlflow.log_metric("train_duration_seconds", train_duration_seconds)

            # Firma del modelo
            signature = None
            if X_sample is not None and y_sample is not None:
                try:
                    preds = model.predict(X_sample)
                    signature = infer_signature(X_sample, preds)
                except Exception as e:
                    logger.warning("No se pudo inferir signature: %s", e)

            # Modelo en MLflow Model Registry
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"hatescan_{run_name}",
            )

            # Vectorizador como artefacto separado
            if vectorizer is not None:
                self._log_vectorizer(vectorizer, run_name)

            run_id = run.info.run_id
            logger.info("Run registrado: %s (run_id=%s)", run_name, run_id)
            return run_id

    # ── Helper: vectorizador ───────────────────────────────────────────────────

    def _log_vectorizer(self, vectorizer: Any, run_name: str) -> None:
        """Guarda el vectorizador como artefacto .joblib en MLflow."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"vectorizer_{run_name}.joblib"
            joblib.dump(vectorizer, path)
            mlflow.log_artifact(str(path), artifact_path="vectorizer")

    # ── Validaciones ───────────────────────────────────────────────────────────

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
                f"{threshold} para '{run_name}'. Revisa el modelo."
            )

    def _validate_no_overfitting(self, f1_train: float, f1_val: float) -> None:
        gap = f1_train - f1_val
        if gap > MAX_OVERFITTING_GAP:
            raise ValueError(
                f"Posible overfitting: gap train/val = {gap:.4f} (> {MAX_OVERFITTING_GAP}). "
                "Revisa regularización o el pipeline de preprocesamiento."
            )

    # ── Consulta ───────────────────────────────────────────────────────────────

    def list_runs(self) -> list[dict]:
        """Lista todos los runs ordenados por F1-macro val (descendente)."""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["metrics.f1_macro DESC"],
        )
        return [
            {
                "run_id":   r.info.run_id,
                "run_name": r.data.tags.get("run_type", "unknown"),
                "status":   r.info.status,
                "f1_macro":  r.data.metrics.get("f1_macro"),
                "precision": r.data.metrics.get("precision"),
                "recall":    r.data.metrics.get("recall"),
            }
            for r in runs
        ]

    def get_best_run(self, metric: str = "f1_macro") -> dict | None:
        """Devuelve el run con el mejor valor de la métrica indicada."""
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
