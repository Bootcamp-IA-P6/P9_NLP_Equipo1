"""
HateScan · src/hatescan/models/predictor.py

Clase de inferencia que:
    1. Carga el transformer guardado en models/artifacts/
    2. Preprocesa los comentarios con hatescan.preprocessing.cleaner
    3. Devuelve el JSON exacto que espera el Streamlit
    4. Persiste en Supabase vía save_hatescan_results()

Uso desde Streamlit:
    from hatescan.models.predictor import HateScanPredictor

    predictor = HateScanPredictor()   # carga el modelo una vez al arrancar
    result = predictor.predict(
        video_url="https://youtube.com/watch?v=...",
        video_id="dQw4w9WgXcQ",
        video_title="Never Gonna Give You Up",
        comments=[{"comment_id": "abc123", "text": "you are stupid"}],
        user_session="user@email.com",
        save_to_db=True,
    )
    # result es el dict JSON listo para Streamlit
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hatescan.models.neural import (
    LABEL_COLUMNS,
    HateScanTransformer,
    HateSpeechDataset,
    TransformerHyperparams,
)

logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "artifacts" / "transformer_roberta_4labels" / "model"

# ── Configuración del modelo ──────────────────────────────────────────────────
MODEL_NAME = "transformer_roberta_4labels"
DEFAULT_THRESHOLD = 0.45
MAX_LENGTH = 128
BATCH_SIZE = 32

# ── Mapeo de etiquetas del modelo → campos JSON/Supabase ─────────────────────
# LABEL_COLUMNS = ['IsAbusive', 'IsHatespeech', 'IsRacist', 'IsProvocative']
#
# IsAbusive    → is_toxic (campo principal) + confidence
# IsHatespeech → categories.is_hatespeech
# IsRacist     → categories.is_racist
# IsProvocative→ categories.is_obscene
# is_threat    → siempre None (no tenemos esta etiqueta en el modelo)
_LABEL_TO_CATEGORY = {
    "IsHatespeech": "is_hatespeech",
    "IsRacist":     "is_racist",
    "IsProvocative": "is_obscene",
    # IsAbusive se trata aparte como is_toxic
}


class HateScanPredictor:
    """
    Predictor de hate speech listo para producción.

    Carga el modelo una sola vez en __init__ para no penalizar
    cada llamada a predict() con el tiempo de carga (~2s en CPU).

    Args:
        threshold: Umbral de clasificación binaria. Default 0.45
                   (valor óptimo encontrado durante evaluación).
        model_path: Ruta al modelo guardado. Default: models/artifacts/.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_path: Optional[Path] = None,
    ) -> None:
        self.threshold = threshold
        self.model_path = model_path or MODEL_PATH

        self._validate_model_path()
        self.tokenizer = self._load_tokenizer()
        self.backbone = self._load_backbone()
        logger.info(
            "HateScanPredictor listo | modelo=%s | threshold=%.2f",
            MODEL_NAME, self.threshold,
        )

    # ── Carga del modelo ──────────────────────────────────────────────────────

    def _validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {self.model_path}.\n"
                "Entrena primero con: uv run python run_training_transformer.py"
            )

    def _load_tokenizer(self) -> AutoTokenizer:
        logger.info("Cargando tokenizer desde %s", self.model_path)
        return AutoTokenizer.from_pretrained(str(self.model_path))

    def _load_backbone(self):
        """
        Reconstruye el HateScanTransformer y carga los pesos guardados.
        Extrae solo el backbone para inferencia (sin FocalLoss).
        """
        logger.info("Cargando pesos del modelo...")
        hp = TransformerHyperparams()
        wrapper = HateScanTransformer(hp)

        safe_path = self.model_path / "model.safetensors"
        bin_path  = self.model_path / "pytorch_model.bin"

        if safe_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safe_path))
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No se encontraron pesos en {self.model_path}. "
                "Busca model.safetensors o pytorch_model.bin."
            )

        wrapper.load_state_dict(state_dict, strict=False)
        backbone = wrapper.backbone
        backbone.eval()
        return backbone

    # ── Preprocesamiento ──────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(texts: list[str]) -> list[str]:
        try:
            from hatescan.preprocessing.cleaner import clean_text
            return [clean_text(t) for t in texts]
        except ImportError:
            import re
            def _basic(t: str) -> str:
                t = t.lower()
                t = re.sub(r"https?://\S+|www\.\S+", "", t)
                t = re.sub(r"@\w+|#\w+", "", t)
                return re.sub(r"\s+", " ", t).strip()
            return [_basic(t) for t in texts]

    # ── Inferencia ────────────────────────────────────────────────────────────

    def _run_inference(self, texts: list[str]) -> np.ndarray:
        """
        Devuelve array (n_comments, NUM_LABELS) con probabilidades sigmoid.
        """
        processed = self._preprocess(texts)
        # Labels=None porque es inferencia pura, no necesitamos targets
        dataset = HateSpeechDataset(processed, self.tokenizer, labels=None, max_length=MAX_LENGTH)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

        all_logits = []
        with torch.no_grad():
            for batch in loader:
                out = self.backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                all_logits.append(out.logits.numpy())

        logits = np.concatenate(all_logits, axis=0)
        return 1 / (1 + np.exp(-logits))  # sigmoid → probabilidades

    # ── Construcción del JSON ─────────────────────────────────────────────────

    def _build_comment_result(
        self,
        comment_id: str,
        text_original: str,
        probs: np.ndarray,          # (NUM_LABELS,) para este comentario
    ) -> dict:
        """
        Construye el dict de un comentario individual en el formato JSON.

        Mapeo de índices de LABEL_COLUMNS:
            0 → IsAbusive    → is_toxic + confidence
            1 → IsHatespeech → categories.is_hatespeech
            2 → IsRacist     → categories.is_racist
            3 → IsProvocative→ categories.is_obscene
        """
        label_idx = {col: i for i, col in enumerate(LABEL_COLUMNS)}

        # IsAbusive como proxy de toxicidad general
        abusive_prob  = float(probs[label_idx["IsAbusive"]])
        is_toxic      = abusive_prob >= self.threshold
        confidence    = round(abusive_prob, 4)

        categories = {
            "is_hatespeech": bool(probs[label_idx["IsHatespeech"]] >= self.threshold),
            "is_racist":     bool(probs[label_idx["IsRacist"]]     >= self.threshold),
            "is_threat":     None,   # no modelado — siempre null
            "is_obscene":    bool(probs[label_idx["IsProvocative"]] >= self.threshold),
        }

        return {
            "comment_id":   comment_id,
            "text_original": text_original,
            "is_toxic":     bool(is_toxic),
            "confidence":   confidence,
            "categories":   categories,
        }

    # ── API pública ───────────────────────────────────────────────────────────

    def predict(
        self,
        video_url: str,
        video_id: str,
        video_title: str,
        comments: list[dict],
        user_session: str = "anonymous",
        save_to_db: bool = True,
    ) -> dict:
        """
        Clasifica una lista de comentarios y devuelve el JSON completo.

        Args:
            video_url:    URL completa del video de YouTube.
            video_id:     ID del video (ej. "dQw4w9WgXcQ").
            video_title:  Título del video.
            comments:     Lista de dicts con keys "comment_id" y "text".
                          Ej: [{"comment_id": "abc123", "text": "you are stupid"}]
            user_session: Identificador de sesión del usuario en Streamlit.
            save_to_db:   Si True, persiste en Supabase automáticamente.

        Returns:
            Dict con la estructura JSON completa lista para Streamlit.
        """
        if not comments:
            logger.warning("predict() llamado con lista de comentarios vacía")
            return self._empty_result(video_url)

        texts = [c["text"] for c in comments]
        ids   = [c.get("comment_id", str(uuid.uuid4())) for c in comments]

        logger.info("Clasificando %d comentarios | threshold=%.2f", len(texts), self.threshold)
        probs = self._run_inference(texts)   # (n_comments, NUM_LABELS)

        comment_results = [
            self._build_comment_result(ids[i], texts[i], probs[i])
            for i in range(len(texts))
        ]

        toxic_count     = sum(1 for c in comment_results if c["is_toxic"])
        non_toxic_count = len(comment_results) - toxic_count

        result = {
            "video_url":       video_url,
            "model_used":      MODEL_NAME,
            "total_comments":  len(comment_results),
            "toxic_count":     toxic_count,
            "non_toxic_count": non_toxic_count,
            "comments":        comment_results,
        }

        if save_to_db:
            self._persist(result, user_session, video_title, video_id)

        return result

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _persist(
        self,
        result: dict,
        user_session: str,
        video_title: str,
        video_id: str,
    ) -> None:
        """Llama al cliente Supabase existente del proyecto."""
        try:
            from src.hatescan.database.database import save_hatescan_results
            success = save_hatescan_results(
                model_output_json=result,
                current_session_user=user_session,
                video_title=video_title,
                video_id=video_id,
            )
            if success:
                logger.info("Resultados persistidos en Supabase correctamente")
            else:
                logger.error("Fallo al persistir en Supabase — revisa las credenciales en .env")
        except ImportError:
            logger.error(
                "hatescan.database.supabase_client no encontrado. "
                "Ajusta la ruta del import según tu estructura de database/."
            )

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(video_url: str) -> dict:
        return {
            "video_url":       video_url,
            "model_used":      MODEL_NAME,
            "total_comments":  0,
            "toxic_count":     0,
            "non_toxic_count": 0,
            "comments":        [],
        }
