#!/usr/bin/env python3
"""Ejecuta el pipeline completo de entrenamiento y registro de modelos.

Este script realiza:
1. Registro de modelos baseline y enhanced en MLflow.
2. Ajuste de hiperparámetros TF-IDF, LogisticRegression y XGBoost.
3. Entrenamiento final de los modelos optimizados.
4. Registro de los modelos afinados en MLflow.
5. Resumen final de métricas y guardado de resultados.

Uso:
    python run_training_pipeline.py
"""

import json
import sys
from pathlib import Path

import spacy
from spacy.cli import download as spacy_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.hatescan.training.baseline import load_dataset, preprocess_text, split_dataset
from src.hatescan.training.hyperparameter_tuning import (
    tune_tfidf,
    tune_logistic_regression,
    tune_xgboost,
    train_tuned_model,
)
from src.hatescan.training.register_models import register_all, register_from_tuned
from src.hatescan.training.trainer import HateScanTrainer

ARTIFACT_DIR = ROOT / "models" / "artifacts"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> None:
    try:
        spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        print(f"spaCy model '{model_name}' no encontrado. Descargando...")
        spacy_download(model_name)
        print(f"spaCy model '{model_name}' descargado.")


def run_baseline_and_enhanced(include_random_forest: bool = True) -> None:
    print("\n=== Registro de modelos baseline y enhanced ===")
    register_all(include_random_forest=include_random_forest, include_tuned=False)
    print("Modelos baseline y enhanced registrados en MLflow.")


def run_hyperparameter_tuning(n_trials_tfidf: int = 30, n_trials_lr: int = 50, n_trials_xgb: int = 50) -> dict:
    print("\n=== Ajuste de hiperparámetros ===")
    df = preprocess_text(load_dataset())
    train_df, _, _ = split_dataset(df)
    corpus_train = train_df["text_processed"]
    y_train = train_df["IsToxic"]

    print("\nTunning TfidfVectorizer...")
    tfidf_result = tune_tfidf(corpus_train, y_train, n_trials=n_trials_tfidf)
    print(f"   Mejor F1-macro: {tfidf_result.best_f1_macro:.4f}")
    print(f"   Params: {tfidf_result.best_params}")

    vectorizer_params = {
        "ngram_min": tfidf_result.best_params["ngram_min"],
        "ngram_max": tfidf_result.best_params["ngram_max"],
        "max_features": tfidf_result.best_params["max_features"],
        "min_df": tfidf_result.best_params["min_df"],
        "max_df": tfidf_result.best_params["max_df"],
    }

    print("\nTunning LogisticRegression...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(
        ngram_range=(vectorizer_params["ngram_min"], vectorizer_params["ngram_max"]),
        max_features=vectorizer_params["max_features"],
        min_df=vectorizer_params["min_df"],
        max_df=vectorizer_params["max_df"],
    )
    X_train_matrix = tfidf.fit_transform(corpus_train)

    lr_result = tune_logistic_regression(X_train_matrix, y_train, n_trials=n_trials_lr)
    print(f"   Mejor F1-macro: {lr_result.best_f1_macro:.4f}")
    print(f"   Params: {lr_result.best_params}")

    print("\nTunning XGBoost...")
    xgb_result = tune_xgboost(X_train_matrix, y_train, n_trials=n_trials_xgb)
    print(f"   Mejor F1-macro: {xgb_result.best_f1_macro:.4f}")
    print(f"   Params: {xgb_result.best_params}")

    print("\nEntrenando modelos finales con los mejores hiperparámetros...")
    tuned_data = train_tuned_model(
        tfidf_result.best_params,
        lr_result.best_params,
        xgb_result.best_params,
        df,
    )
    print("Modelos optimizados entrenados y guardados en models/artifacts/baseline_tuned.")

    return {
        "tfidf": {
            "best_f1_macro": tfidf_result.best_f1_macro,
            "best_params": tfidf_result.best_params,
        },
        "logistic_regression": {
            "best_f1_macro": lr_result.best_f1_macro,
            "best_params": lr_result.best_params,
        },
        "xgboost": {
            "best_f1_macro": xgb_result.best_f1_macro,
            "best_params": xgb_result.best_params,
        },
    }


def register_tuned_models() -> None:
    print("\n=== Registro de modelos optimizados en MLflow ===")
    register_from_tuned()
    print("Modelos afinados registrados en MLflow.")


def save_summary(summary: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Resumen guardado en {SUMMARY_PATH}")


def show_mlflow_summary(trainer: HateScanTrainer) -> dict:
    runs = trainer.list_runs()
    best_run = trainer.get_best_run()

    print("\n=== Resumen de experimentos en MLflow ===")
    for run in runs:
        print(
            f"- {run['run_name']}: F1_macro={run['f1_macro']:.4f} | "
            f"precision={run['precision']:.4f} | recall={run['recall']:.4f} | roc_auc={run['roc_auc']:.4f}"
        )

    if best_run:
        print(f"\nMejor run: {best_run['run_name']} (run_id={best_run['run_id']})")
        print(f"   Métrica principal: f1_macro={best_run['metrics']['f1_macro']:.4f}")

    return {
        "best_run": best_run,
        "all_runs": runs,
    }


def main() -> None:
    ensure_spacy_model()
    trainer = HateScanTrainer()

    run_baseline_and_enhanced(include_random_forest=True)
    tuning_info = run_hyperparameter_tuning(n_trials_tfidf=30, n_trials_lr=50, n_trials_xgb=50)
    register_tuned_models()

    mlflow_summary = show_mlflow_summary(trainer)
    save_summary({"tuning_info": tuning_info, "mlflow_summary": mlflow_summary})

    print("\n=== Pipeline completo finalizado ===")


if __name__ == "__main__":
    main()
