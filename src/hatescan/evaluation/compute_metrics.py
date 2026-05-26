"""
HateScan · src/hatescan/evaluation/compute_metrics.py

Función compute_metrics compatible con HuggingFace Trainer para
clasificación multilabel con umbral configurable.

Por qué no usamos el compute_metrics por defecto de HF:
    El default calcula accuracy, que es inútil para multilabel desbalanceado.
    Necesitamos F1-macro (métrica principal del proyecto) con umbral ajustado
    a 0.3 para capturar las clases raras.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hatescan.models.neural import LABEL_COLUMNS


def make_compute_metrics(threshold: float = 0.3):
    """
    Factory que devuelve un compute_metrics listo para pasar a Trainer.

    El threshold se configura en TransformerHyperparams.cls_threshold
    y se pasa aquí desde train_transformer.py.

    Args:
        threshold: Umbral de clasificación binaria sobre sigmoid(logits).
                   Default 0.3 porque las etiquetas raras (IsRacist, IsHatespeech)
                   raramente superan 0.5 con 700 ejemplos de entrenamiento.

    Returns:
        Función compute_metrics(eval_pred) → dict[str, float]
    """

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))          # sigmoid manual (no torch aquí)
        preds = (probs >= threshold).astype(int)

        # ── Métricas globales (las que loguea MLflow como principales) ────────
        f1_macro   = f1_score(labels, preds, average="macro",     zero_division=0)
        f1_micro   = f1_score(labels, preds, average="micro",     zero_division=0)
        precision  = precision_score(labels, preds, average="macro", zero_division=0)
        recall     = recall_score(labels, preds, average="macro",    zero_division=0)

        # ROC-AUC requiere al menos 2 clases activas en el batch
        try:
            roc_auc = roc_auc_score(labels, probs, average="macro")
        except ValueError:
            roc_auc = 0.0

        # ── F1 por etiqueta (para diagnóstico en MLflow) ──────────────────────
        f1_per_label = f1_score(labels, preds, average=None, zero_division=0)
        per_label_metrics = {
            f"f1_{col.lower()}": round(float(f1_per_label[i]), 4)
            for i, col in enumerate(LABEL_COLUMNS)
        }

        metrics = {
            "f1_macro":  round(f1_macro,  4),
            "f1_micro":  round(f1_micro,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "roc_auc":   round(roc_auc,   4),
            **per_label_metrics,
        }
        return metrics

    return compute_metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Imprime el classification report por etiqueta para análisis manual."""
    print("\n── Classification Report por etiqueta ──────────────────────────")
    print(classification_report(y_true, y_pred, target_names=LABEL_COLUMNS, zero_division=0))
