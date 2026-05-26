"""
HateScan · src/hatescan/models/neural.py

Modelo transformer para clasificación multilabel de hate speech.
Target: transformer_roberta_4labels  →  4 etiquetas modelables del dataset:
    IsAbusive · IsHatespeech · IsRacist · IsProvocative

Por qué estas 4 y no las 12:
    Las otras (IsNationalist=8, IsSexist=1, IsHomophobic=0, IsRadicalism=0)
    no tienen positivos suficientes para que ningún modelo aprenda nada útil.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ── Etiquetas modelables ──────────────────────────────────────────────────────
LABEL_COLUMNS: list[str] = [
    "IsAbusive",
    "IsHatespeech",
    "IsRacist",
    "IsProvocative",
]
NUM_LABELS = len(LABEL_COLUMNS)

# ── Modelo base ───────────────────────────────────────────────────────────────
# twitter-roberta-base-hate está preentrenado en tweets de odio en inglés,
# exactamente el dominio de YouTube comments → ventaja de dominio frente a
# bert-base-uncased genérico.
DEFAULT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate"


# ── Hiperparámetros de fine-tuning ────────────────────────────────────────────
@dataclass
class TransformerHyperparams:
    """
    Todos los hiperparámetros en un único dataclass para que sean
    reproducibles y logueables en MLflow sin magia.

    Justificación de cada valor:
        num_train_epochs=2   → con ~700 ejemplos la época 3 ya memoriza
        learning_rate=2e-5   → default HF (5e-5) es agresivo para multilabel escaso
        warmup_ratio=0.1     → protege los pesos preentrenados los primeros pasos
        weight_decay=0.01    → L2 suave, casi siempre ayuda en fine-tuning
        batch_size=16        → batches grandes estabilizan gradiente con pocos datos
        cls_threshold=0.3    → etiquetas raras nunca alcanzan 0.5; bajar umbral
                               mejora recall sin cambiar el modelo
        focal_loss=True      → penaliza más los ejemplos difíciles (clases raras)
        focal_gamma=2.0      → valor estándar de la literatura focal loss
    """

    model_name: str = DEFAULT_MODEL_NAME
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    max_length: int = 128
    cls_threshold: float = 0.3
    focal_loss: bool = True
    focal_gamma: float = 2.0
    # Pesos por clase: se calculan automáticamente en train_transformer.py
    # a partir de la frecuencia inversa de cada etiqueta.
    class_weights: Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Para mlflow.log_params()."""
        return {
            "model_name": self.model_name,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "train_batch_size": self.per_device_train_batch_size,
            "max_length": self.max_length,
            "cls_threshold": self.cls_threshold,
            "focal_loss": self.focal_loss,
            "focal_gamma": self.focal_gamma,
        }


# ── Dataset ───────────────────────────────────────────────────────────────────
class HateSpeechDataset(Dataset):
    """
    Dataset PyTorch para clasificación multilabel.

    Recibe textos ya preprocesados (limpios, sin stopwords, lematizados)
    y los tokeniza on-the-fly para no guardar tensores en disco.

    Args:
        texts:      Lista de textos preprocesados.
        labels:     Array numpy (n_samples, NUM_LABELS) con 0/1 por etiqueta.
                    Puede ser None en inferencia.
        tokenizer:  Tokenizer de HuggingFace ya cargado.
        max_length: Longitud máxima de tokens (default 128 — comentarios de
                    YouTube rara vez superan 100 tokens tras preprocesamiento).
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: AutoTokenizer,
        labels: Optional[np.ndarray] = None,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            # float32 requerido por BCEWithLogitsLoss / FocalLoss
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multilabel con clases desbalanceadas.

    Con etiquetas como IsRacist (125/1000 positivos) la BCE estándar está
    dominada por los negativos. Focal Loss añade un factor (1-p)^gamma que
    reduce el peso de los ejemplos fáciles y fuerza al modelo a aprender
    los difíciles (los positivos de clases raras).

    Referencia: Lin et al., 2017 — "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        # p_t = prob del target real (1 si target=1, 1-prob si target=0)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ── Modelo con pérdida custom ─────────────────────────────────────────────────
class HateScanTransformer(nn.Module):
    """
    Wrapper sobre AutoModelForSequenceClassification que inyecta
    FocalLoss o BCEWithLogitsLoss ponderada según los hiperparámetros.

    HuggingFace Trainer espera que el modelo devuelva una tupla
    (loss, logits) cuando recibe labels en el batch — este wrapper
    lo garantiza independientemente del modelo base.
    """

    def __init__(self, hyperparams: TransformerHyperparams) -> None:
        super().__init__()
        self.hp = hyperparams
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            hyperparams.model_name,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,  # la cabeza de clasificación se reinicia
        )

        # Pérdida
        pos_weight = None
        if hyperparams.class_weights:
            pos_weight = torch.tensor(hyperparams.class_weights, dtype=torch.float32)

        if hyperparams.focal_loss:
            self.loss_fn = FocalLoss(gamma=hyperparams.focal_gamma, pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        logger.info(
            "HateScanTransformer inicializado: %s | pérdida=%s | etiquetas=%s",
            hyperparams.model_name,
            "FocalLoss" if hyperparams.focal_loss else "BCEWithLogitsLoss",
            LABEL_COLUMNS,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        # Pasamos por el backbone sin calcular su propia pérdida
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (batch, NUM_LABELS)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        # Trainer espera (loss, logits) o ModelOutput con .loss y .logits
        return (loss, logits) if loss is not None else (logits,)
