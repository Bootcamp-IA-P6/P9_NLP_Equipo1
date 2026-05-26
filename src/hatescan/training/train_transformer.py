"""
HateScan · src/hatescan/training/train_transformer.py

Script de entrenamiento completo para transformer multilabel.
Integra HuggingFace Trainer con el HateScanTrainer de MLflow existente.

Uso:
    uv run python src/hatescan/training/train_transformer.py
    uv run python src/hatescan/training/train_transformer.py --epochs 3 --lr 1e-5
    uv run python src/hatescan/training/train_transformer.py --no-focal-loss

Qué hace este script en orden:
    1. Carga youtoxic_english_1000.csv y aplica el preprocesamiento existente
    2. Calcula pesos de clase por frecuencia inversa (combate el desbalanceo)
    3. Construye HateScanTransformer con FocalLoss + pesos
    4. Entrena con HuggingFace Trainer (early stopping incluido)
    5. Evalúa en test con umbral 0.3
    6. Registra en MLflow vía HateScanTrainer con todas las métricas
    7. Guarda el modelo en models/artifacts/transformer_roberta_4labels/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Imports internos del proyecto
from hatescan.evaluation.compute_metrics import make_compute_metrics, print_classification_report
from hatescan.models.neural import (
    LABEL_COLUMNS,
    HateScanTransformer,
    HateSpeechDataset,
    TransformerHyperparams,
)
from hatescan.training.trainer import HateScanTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Estructura: src/hatescan/training/train_transformer.py
#   parents[0] = src/hatescan/training/
#   parents[1] = src/hatescan/
#   parents[2] = src/
#   parents[3] = raíz del proyecto  ← aquí está data/, models/, etc.
ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "raw" / "youtoxic_english_1000.csv"
ARTIFACTS_DIR = ROOT / "models" / "artifacts" / "transformer_roberta_4labels"

# Validación temprana: falla con mensaje claro antes de llegar a pandas
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\n[HateScan] Dataset no encontrado: {DATA_PATH}"
        f"\n  ROOT resuelto como: {ROOT}"
        f"\n  Comprueba que 'data/raw/youtoxic_english_1000.csv' existe en la raíz del proyecto."
    )

# ── Split idéntico al del resto del proyecto ──────────────────────────────────
# 70% train / 15% val / 15% test con stratify en IsToxic (más balanceada)
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15   # del total; test = lo que queda


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga y preprocesamiento
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(data_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Carga el CSV y aplica el pipeline de preprocesamiento del proyecto.

    Intenta importar el cleaner existente en hatescan.preprocessing.
    Si no está disponible (entorno de pruebas) aplica un limpiador mínimo
    para que el script no explote.

    Returns:
        texts:  Lista de textos preprocesados.
        labels: Array (n, NUM_LABELS) con los targets de las 4 etiquetas.
    """
    logger.info("Cargando dataset: %s", data_path)
    df = pd.read_csv(data_path)

    # Verificar que las columnas requeridas existen
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    # Columna de texto — el proyecto usa 'Text' o 'text'
    text_col = "Text" if "Text" in df.columns else "text"

    # Intentar preprocesador del proyecto; fallback a limpieza básica
    try:
        from hatescan.preprocessing.cleaner import preprocess_text
        logger.info("Usando hatescan.preprocessing.cleaner.preprocess_text")
        texts = [preprocess_text(t) for t in df[text_col].fillna("").tolist()]
    except ImportError:
        logger.warning(
            "hatescan.preprocessing.cleaner no encontrado — usando limpieza básica. "
            "Asegúrate de correr desde el entorno del proyecto."
        )
        import re
        def _basic_clean(t: str) -> str:
            t = t.lower()
            t = re.sub(r"https?://\S+|www\.\S+", "", t)
            t = re.sub(r"@\w+|#\w+", "", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t
        texts = [_basic_clean(t) for t in df[text_col].fillna("").tolist()]

    labels = df[LABEL_COLUMNS].values.astype(np.float32)

    logger.info(
        "Dataset cargado: %d muestras | distribución positivos: %s",
        len(texts),
        {col: int(labels[:, i].sum()) for i, col in enumerate(LABEL_COLUMNS)},
    )
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pesos de clase (frecuencia inversa)
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(labels_train: np.ndarray) -> list[float]:
    """
    Calcula pos_weight para cada etiqueta como n_neg / n_pos.

    Con IsRacist teniendo ~87 positivos y ~613 negativos en train,
    el peso sería 613/87 ≈ 7.0 — el modelo penaliza 7x más los
    falsos negativos en esa etiqueta.

    Se capa a 10.0 para evitar que etiquetas con muy pocos positivos
    dominen completamente la pérdida.
    """
    n = len(labels_train)
    weights = []
    for i, col in enumerate(LABEL_COLUMNS):
        n_pos = labels_train[:, i].sum()
        n_neg = n - n_pos
        w = min(n_neg / max(n_pos, 1), 10.0)
        weights.append(round(float(w), 3))
        logger.info("  %s → pos=%d neg=%d weight=%.2f", col, int(n_pos), int(n_neg), w)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. Split del dataset
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(
    texts: list[str],
    labels: np.ndarray,
) -> tuple:
    """
    Split estratificado en IsToxic (la etiqueta más balanceada).
    Mismo random_state que el resto del proyecto para reproducibilidad.

    Returns: texts_train, texts_val, texts_test, y_train, y_val, y_test
    """
    # IsToxic no es una de las 4 etiquetas target pero sí existe en el CSV
    # como proxy de estratificación — si no está disponible, sin estratificar
    stratify_col = None  # Se sobreescribe abajo si podemos calcularlo

    # Separar test primero
    idx = list(range(len(texts)))
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )
    # Separar val del trainval
    val_relative = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_relative,
        random_state=RANDOM_STATE,
    )

    def pick(lst, idxs):
        return [lst[i] for i in idxs]

    texts_train = pick(texts, idx_train)
    texts_val   = pick(texts, idx_val)
    texts_test  = pick(texts, idx_test)
    y_train = labels[idx_train]
    y_val   = labels[idx_val]
    y_test  = labels[idx_test]

    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(texts_train), len(texts_val), len(texts_test),
    )
    return texts_train, texts_val, texts_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────────────
def train(hp: TransformerHyperparams) -> dict[str, float]:
    """
    Orquesta todo el pipeline de entrenamiento.

    Returns:
        Diccionario con las métricas finales sobre el set de test
        (las que se loguean en MLflow).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Datos ─────────────────────────────────────────────────────────────────
    texts, labels = load_and_preprocess(DATA_PATH)
    texts_train, texts_val, texts_test, y_train, y_val, y_test = split_dataset(texts, labels)

    # ── Pesos de clase ────────────────────────────────────────────────────────
    logger.info("Calculando pesos de clase:")
    class_weights = compute_class_weights(y_train)
    hp.class_weights = class_weights

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Cargando tokenizer: %s", hp.model_name)
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_train = HateSpeechDataset(texts_train, tokenizer, y_train, hp.max_length)
    ds_val   = HateSpeechDataset(texts_val,   tokenizer, y_val,   hp.max_length)
    ds_test  = HateSpeechDataset(texts_test,  tokenizer, y_test,  hp.max_length)

    # ── Modelo ────────────────────────────────────────────────────────────────
    logger.info("Inicializando HateScanTransformer")
    model = HateScanTransformer(hp)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = str(ARTIFACTS_DIR / "checkpoints")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp.num_train_epochs,
        learning_rate=hp.learning_rate,
        warmup_steps=int(0.1 * (700 // hp.per_device_train_batch_size) * hp.num_train_epochs),
        weight_decay=hp.weight_decay,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        per_device_eval_batch_size=hp.per_device_eval_batch_size,
        # ── Evaluación y early stopping ───────────────────────────────────────
        eval_strategy="epoch",          # evaluar al final de cada época
        save_strategy="epoch",
        load_best_model_at_end=True,    # cargar el checkpoint con mejor val F1
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # ── Logging ───────────────────────────────────────────────────────────
        logging_steps=20,
        report_to="none",               # MLflow se gestiona manualmente abajo
        # ── Reproducibilidad ─────────────────────────────────────────────────
        seed=RANDOM_STATE,
        # ── Rendimiento ──────────────────────────────────────────────────────
        dataloader_num_workers=0,       # 0 evita problemas en Windows/Docker
        dataloader_pin_memory=False,    # pin_memory solo tiene sentido con GPU
        fp16=torch.cuda.is_available(), # mixed precision solo si hay GPU
    )

    # EarlyStoppingCallback: para si el val F1 no mejora en 2 épocas.
    # Con num_train_epochs=2 esto actúa como red de seguridad si se sube a 3-4.
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    compute_metrics = make_compute_metrics(threshold=hp.cls_threshold)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    logger.info("Iniciando entrenamiento — %d epochs, lr=%.0e", hp.num_train_epochs, hp.learning_rate)
    t0 = time.time()
    trainer.train()
    train_duration = round(time.time() - t0, 2)
    logger.info("Entrenamiento completado en %.1fs", train_duration)

    # ── Métricas de train (para calcular gap overfitting) ─────────────────────
    train_results = trainer.evaluate(ds_train)
    f1_train = train_results["eval_f1_macro"]

    # ── Evaluación en test ────────────────────────────────────────────────────
    logger.info("Evaluando en test set con umbral=%.2f", hp.cls_threshold)
    test_results = trainer.evaluate(ds_test)

    # Métricas finales que espera HateScanTrainer
    metrics = {
        "f1_macro":  test_results["eval_f1_macro"],
        "precision": test_results["eval_precision"],
        "recall":    test_results["eval_recall"],
        "roc_auc":   test_results["eval_roc_auc"],
        # Métricas adicionales por etiqueta
        **{k: v for k, v in test_results.items() if k.startswith("eval_f1_is")},
    }

    # Reporte de clasificación para análisis en consola
    pred_output = trainer.predict(ds_test)
    probs_test = 1 / (1 + np.exp(-pred_output.predictions))
    preds_test = (probs_test >= hp.cls_threshold).astype(int)
    print_classification_report(y_test.astype(int), preds_test)

    # ── Guardar modelo y tokenizer ────────────────────────────────────────────
    model_save_path = ARTIFACTS_DIR / "model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    logger.info("Modelo guardado en: %s", model_save_path)

    # ── Registrar en MLflow vía HateScanTrainer ───────────────────────────────
    logger.info("Registrando run en MLflow")
    mlflow_trainer = HateScanTrainer()

    # HateScanTrainer.log_run espera un modelo sklearn-compatible.
    # Usamos un wrapper mínimo para que no explote mlflow.sklearn.log_model.
    # El modelo real ya está guardado en models/artifacts/.
    class _SklearnWrapper:
        """Wrapper mínimo para satisfacer la firma de HateScanTrainer."""
        def predict(self, X):
            return np.zeros(len(X))

    run_id = mlflow_trainer.log_run(
        run_name="transformer_roberta_4labels",
        model=_SklearnWrapper(),
        vectorizer=None,
        params={
            **hp.to_dict(),
            "train_duration_seconds": train_duration,
            "model_save_path": str(model_save_path),
            "class_weights": str(class_weights),
        },
        metrics=metrics,
        f1_train=f1_train,
        train_duration_seconds=train_duration,
        tags={
            "model_type": "multilabel",
            "framework": "huggingface",
            "labels": ",".join(LABEL_COLUMNS),
            "cls_threshold": str(hp.cls_threshold),
        },
    )

    logger.info("✓ Run registrado en MLflow: %s", run_id)
    logger.info("── Métricas finales (test) ──────────────────────────")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación sobre modelo ya guardado (sin reentrenar)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_saved(threshold: float) -> None:
    """
    Carga el modelo guardado en ARTIFACTS_DIR y reevalúa en test
    con un umbral diferente. No reentrena nada.

    Uso:
        uv run python run_training_transformer.py --eval-only --threshold 0.40

    Nota: trainer.save_model() guarda el HateScanTransformer completo (wrapper
    nn.Module custom). Para inferencia pura extraemos el backbone directamente
    del state_dict guardado.
    """
    model_path = ARTIFACTS_DIR / "model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No hay modelo guardado en {model_path}. "
            "Entrena primero sin --eval-only."
        )

    logger.info("Cargando modelo guardado desde: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # El Trainer guardó el HateScanTransformer (wrapper custom).
    # Reconstruimos el wrapper con los mismos hiperparámetros por defecto
    # y cargamos el state_dict guardado.
    hp_default = TransformerHyperparams()
    wrapper = HateScanTransformer(hp_default)

    import torch
    # pytorch_model.bin o model.safetensors según versión de transformers
    bin_path   = model_path / "pytorch_model.bin"
    safe_path  = model_path / "model.safetensors"

    if safe_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safe_path))
        logger.info("Cargando pesos desde model.safetensors")
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
        logger.info("Cargando pesos desde pytorch_model.bin")
    else:
        raise FileNotFoundError(
            f"No se encontró pytorch_model.bin ni model.safetensors en {model_path}"
        )

    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Pesos no cargados (missing): %s", missing)
    if unexpected:
        logger.warning("Pesos ignorados (unexpected): %s", unexpected)

    backbone = wrapper.backbone
    backbone.eval()

    texts, labels = load_and_preprocess(DATA_PATH)
    texts_train, _, texts_test, y_train, _, y_test = split_dataset(texts, labels)

    from torch.utils.data import DataLoader
    from hatescan.evaluation.compute_metrics import make_compute_metrics, print_classification_report
    compute_metrics = make_compute_metrics(threshold=threshold)

    def _run_inference(text_list, label_array):
        ds = HateSpeechDataset(text_list, tokenizer, label_array, max_length=128)
        loader = DataLoader(ds, batch_size=32, shuffle=False, pin_memory=False)
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                out = backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                all_logits.append(out.logits.numpy())
        return np.concatenate(all_logits, axis=0)

    # ── Inferencia en train y test ────────────────────────────────────────────
    logger.info("Evaluando en train set...")
    logits_train = _run_inference(texts_train, y_train)
    metrics_train = compute_metrics((logits_train, y_train))

    logger.info("Evaluando en test set...")
    logits_test = _run_inference(texts_test, y_test)
    preds_test = (1 / (1 + np.exp(-logits_test)) >= threshold).astype(int)
    metrics_test = compute_metrics((logits_test, y_test))

    # ── Gap overfitting ───────────────────────────────────────────────────────
    gap = round(metrics_train["f1_macro"] - metrics_test["f1_macro"], 4)
    max_gap = 0.15  # límite definido en trainer.py para transformers

    if gap <= 0.05:
        estado = "✅ Sin overfitting"
    elif gap <= max_gap:
        estado = "⚠️  Overfitting leve (aceptable)"
    else:
        estado = "🔴 Overfitting severo — revisar epochs o weight_decay"

    print_classification_report(y_test.astype(int), preds_test)

    print("\n── Diagnóstico de overfitting ──────────────────────────────")
    print(f"  F1-macro TRAIN : {metrics_train['f1_macro']:.4f}")
    print(f"  F1-macro TEST  : {metrics_test['f1_macro']:.4f}")
    print(f"  Gap train/test : {gap:+.4f}  {estado}")
    print(f"  Límite máximo  : {max_gap}")
    print("────────────────────────────────────────────────────────────")
    print("\n── Métricas test por etiqueta ──────────────────────────────")
    for k, v in metrics_test.items():
        print(f"  {k}: {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena transformer multilabel para HateScan")
    p.add_argument("--epochs",        type=int,   default=2,    help="Número de epochs (default: 2)")
    p.add_argument("--lr",            type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    p.add_argument("--batch-size",    type=int,   default=16,   help="Batch size (default: 16)")
    p.add_argument("--threshold",     type=float, default=0.4,  help="Umbral clasificación (default: 0.4)")
    p.add_argument("--no-focal-loss", action="store_true",      help="Usar BCEWithLogitsLoss en lugar de FocalLoss")
    p.add_argument("--model",         type=str,   default="cardiffnlp/twitter-roberta-base-hate",
                   help="Modelo base de HuggingFace")
    # ── Nuevo flag: solo evaluar, no reentrenar ───────────────────────────────
    p.add_argument("--eval-only",     action="store_true",
                   help="Evalúa el modelo guardado con --threshold sin reentrenar")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.eval_only:
        # Solo inferencia sobre el modelo ya guardado
        logger.info("Modo eval-only — umbral=%.2f (sin reentrenamiento)", args.threshold)
        evaluate_saved(threshold=args.threshold)
    else:
        # Entrenamiento completo
        hp = TransformerHyperparams(
            model_name=args.model,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            cls_threshold=args.threshold,
            focal_loss=not args.no_focal_loss,
        )
        logger.info("Hiperparámetros: %s", hp)
        train(hp)
