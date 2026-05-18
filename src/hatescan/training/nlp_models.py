"""
HateScan · ISSUE-05
src/hatescan/training/nlp_models.py

Pipeline NLP clásico:
  - TF-IDF (ngram_range=(1,2), max_features=10000)
  - Baseline: Logistic Regression
  - XGBoost con scale_pos_weight
  - Random Forest
  - Split 70/15/15 con StratifiedKFold
  - Serialización en models/artifacts/
  - SIN MLflow (el registro lo hace DS #2 en ISSUE-06)
"""

import time
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ── Split 70 / 15 / 15 ────────────────────────────────────────────────────────

def make_splits(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """
    Split estratificado 70 / 15 / 15.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Primero 70 train / 30 temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    # Luego 50/50 del 30% → 15 val / 15 test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    logger.info(
        "Splits → train: %d | val: %d | test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Métricas ───────────────────────────────────────────────────────────────────

def compute_metrics(model, X, y, label: str = "") -> dict:
    """Calcula F1-macro, Precision, Recall y ROC-AUC."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "f1_macro":  round(f1_score(y, y_pred, average="macro"), 4),
        "precision": round(precision_score(y, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, average="macro", zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_proba), 4),
    }
    if label:
        logger.info("[%s] %s", label, metrics)
    return metrics


# ── TF-IDF ─────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer() -> TfidfVectorizer:
    """TF-IDF con configuración del plan: ngram(1,2), max_features=10000."""
    return TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)


# ── Modelos ────────────────────────────────────────────────────────────────────

def train_baseline_lr(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    # scale_pos_weight compensa el desbalanceo de clases
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2) if pos > 0 else 1.0
    logger.info("XGBoost scale_pos_weight: %.2f", spw)

    model = XGBClassifier(
        n_estimators=200,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_nlp_pipeline(df: pd.DataFrame, text_col: str = "text_processed", target_col: str = "IsToxic"):
    """
    Ejecuta el pipeline completo: vectorización → splits → entrenamiento → métricas.

    Args:
        df:         DataFrame con texto preprocesado y etiqueta.
        text_col:   Columna con texto ya limpio y lematizado.
        target_col: Columna target binaria (0/1).

    Returns:
        dict con modelos, vectorizador, splits y métricas.
    """
    logger.info("=== Pipeline NLP HateScan ===")

    texts = df[text_col].fillna("").to_numpy()
    y     = df[target_col].to_numpy()

    # 1. Vectorizador — fit SOLO en train (anti data-leakage)
    tfidf_vectorizer = build_tfidf_vectorizer()

    # Primero hacemos los splits sobre texto crudo para evitar leakage
    X_text_train, X_text_val, X_text_test, y_train, y_val, y_test = make_splits(texts, y)

    # Fit del vectorizador SOLO en train
    X_train = tfidf_vectorizer.fit_transform(X_text_train)
    X_val   = tfidf_vectorizer.transform(X_text_val)
    X_test  = tfidf_vectorizer.transform(X_text_test)

    logger.info("TF-IDF features: %d", X_train.shape[1])

    results = {}

    # 2. Baseline LR
    logger.info("--- Entrenando baseline_lr ---")
    t0 = time.time()
    baseline_lr = train_baseline_lr(X_train, y_train)
    results["baseline_lr"] = {
        "model": baseline_lr,
        "train_secs": round(time.time() - t0, 2),
        "val_metrics":   compute_metrics(baseline_lr, X_val,   y_val,   "LR val"),
        "train_metrics": compute_metrics(baseline_lr, X_train, y_train, "LR train"),
        "test_metrics":  compute_metrics(baseline_lr, X_test,  y_test,  "LR test"),
    }

    # 3. XGBoost
    logger.info("--- Entrenando xgboost ---")
    t0 = time.time()
    xgboost_model = train_xgboost(X_train, y_train)
    results["xgboost"] = {
        "model": xgboost_model,
        "train_secs": round(time.time() - t0, 2),
        "val_metrics":   compute_metrics(xgboost_model, X_val,   y_val,   "XGB val"),
        "train_metrics": compute_metrics(xgboost_model, X_train, y_train, "XGB train"),
        "test_metrics":  compute_metrics(xgboost_model, X_test,  y_test,  "XGB test"),
    }

    # 4. Random Forest
    logger.info("--- Entrenando random_forest ---")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train)
    results["random_forest"] = {
        "model": rf_model,
        "train_secs": round(time.time() - t0, 2),
        "val_metrics":   compute_metrics(rf_model, X_val,   y_val,   "RF val"),
        "train_metrics": compute_metrics(rf_model, X_train, y_train, "RF train"),
        "test_metrics":  compute_metrics(rf_model, X_test,  y_test,  "RF test"),
    }

    # 5. Serialización en models/artifacts/
    joblib.dump(tfidf_vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    for name, data in results.items():
        joblib.dump(data["model"], ARTIFACTS_DIR / f"{name}.joblib")
        logger.info("Serializado: models/artifacts/%s.joblib", name)

    # 6. Resumen
    logger.info("\n=== RESUMEN (val set) ===")
    for name, data in results.items():
        m = data["val_metrics"]
        gap = data["train_metrics"]["f1_macro"] - m["f1_macro"]
        logger.info(
            "%-20s F1=%.4f  P=%.4f  R=%.4f  gap=%.4f %s",
            name, m["f1_macro"], m["precision"], m["recall"], gap,
            "⚠️ OVERFITTING" if gap > 0.05 else "✅"
        )

    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "baseline_lr":   results["baseline_lr"]["model"],
        "xgboost_model": results["xgboost"]["model"],
        "rf_model":      results["random_forest"]["model"],
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "results": results,
    }
