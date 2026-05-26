import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from src.hatescan.preprocessing import TextTokenizer, extract_case_features
from src.hatescan.training.baseline import run_training as run_baseline_training
from src.hatescan.training.enhanced import (
    extract_text_features,
    load_dataset,
    preprocess_text,
    run_training as run_enhanced_training,
)


def compare_pipelines():
    """Train both pipelines and compare metrics."""
    print("=" * 70)
    print("ENTRENANDO BASELINE (Text only)...")
    print("=" * 70)
    baseline_results = run_baseline_training(include_random_forest=False)

    print("\n" + "=" * 70)
    print("ENTRENANDO ENHANCED (Text + Features)...")
    print("=" * 70)
    enhanced_results, enh_vectorizer, enh_scaler, enh_lr, enh_xgb = run_enhanced_training(
        include_random_forest=False
    )

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MÉTRICAS (TEST SET)")
    print("=" * 70)

    metrics_list = ["f1_macro", "precision_macro", "recall_macro", "roc_auc"]

    for metric in metrics_list:
        print(f"\n{metric.upper()}:")
        baseline_val = getattr(baseline_results["logistic_regression"], metric)
        enhanced_val = getattr(enhanced_results["logistic_regression"], metric)
        diff = enhanced_val - baseline_val
        improvement = (diff / baseline_val * 100) if baseline_val != 0 else 0
        print(f"  Baseline:  {baseline_val:.4f}")
        print(f"  Enhanced:  {enhanced_val:.4f}")
        print(f"  Diff:      {diff:+.4f} ({improvement:+.1f}%)")

    # Test sentence
    test_sentence = "You disgust me, you fucking black bitch. What the hell are you going to do with Spain? You're a fucking African whore. You don't belong here."

    print("\n" + "=" * 70)
    print("TEST CON FRASE DE HATE SPEECH")
    print("=" * 70)
    print(f"Frase: {test_sentence[:80]}...")

    # Load baseline models
    baseline_vectorizer = joblib.load("models/artifacts/baseline/tfidf_vectorizer.joblib")
    baseline_lr = joblib.load("models/artifacts/baseline/logistic_regression.joblib")
    baseline_xgb = joblib.load("models/artifacts/baseline/xgboost_classifier.joblib")

    # Prepare test sentence for baseline
    tokenizer = TextTokenizer()
    text_processed = tokenizer.process(test_sentence)
    X_test_baseline = baseline_vectorizer.transform([text_processed])

    # Predictions - baseline
    baseline_lr_pred = baseline_lr.predict([X_test_baseline.toarray()[0]])[0]
    baseline_lr_proba = baseline_lr.predict_proba([X_test_baseline.toarray()[0]])[0]

    baseline_xgb_pred = baseline_xgb.predict([X_test_baseline.toarray()[0]])[0]
    baseline_xgb_proba = baseline_xgb.predict_proba([X_test_baseline.toarray()[0]])[0]

    # Prepare test sentence for enhanced
    df_test = pd.DataFrame({"Text": [test_sentence], "IsToxic": [0]})
    df_test = preprocess_text(df_test)
    df_test = extract_text_features(df_test)

    text_processed_enh = df_test["text_processed"].values[0]
    X_test_enh_tfidf = enh_vectorizer.transform([text_processed_enh])

    extra_features_cols = [
        "text_length",
        "word_count",
        "has_uppercase",
        "all_caps_ratio",
        "exclamation_count",
        "question_count",
    ]
    X_test_enh_extra = df_test[extra_features_cols].values
    X_test_enh_extra = enh_scaler.transform(X_test_enh_extra)
    X_test_enh = hstack([X_test_enh_tfidf, X_test_enh_extra])

    # Predictions - enhanced
    enh_lr_pred = enh_lr.predict([X_test_enh.toarray()[0]])[0]
    enh_lr_proba = enh_lr.predict_proba([X_test_enh.toarray()[0]])[0]

    enh_xgb_pred = enh_xgb.predict([X_test_enh.toarray()[0]])[0]
    enh_xgb_proba = enh_xgb.predict_proba([X_test_enh.toarray()[0]])[0]

    print("\n📊 BASELINE PIPELINE (Text Only)")
    print("-" * 70)
    print(f"Logistic Regression:")
    print(f"  Predicción: {'TÓXICO' if baseline_lr_pred == 1 else 'NO TÓXICO'} (pred={baseline_lr_pred})")
    print(f"  Probabilidad: No Tóxico={baseline_lr_proba[0]:.4f}, Tóxico={baseline_lr_proba[1]:.4f}")

    print(f"XGBoost:")
    print(f"  Predicción: {'TÓXICO' if baseline_xgb_pred == 1 else 'NO TÓXICO'} (pred={baseline_xgb_pred})")
    print(f"  Probabilidad: No Tóxico={baseline_xgb_proba[0]:.4f}, Tóxico={baseline_xgb_proba[1]:.4f}")

    print("\n📊 ENHANCED PIPELINE (Text + Features)")
    print("-" * 70)
    print(f"Logistic Regression:")
    print(f"  Predicción: {'TÓXICO' if enh_lr_pred == 1 else 'NO TÓXICO'} (pred={enh_lr_pred})")
    print(f"  Probabilidad: No Tóxico={enh_lr_proba[0]:.4f}, Tóxico={enh_lr_proba[1]:.4f}")

    print(f"XGBoost:")
    print(f"  Predicción: {'TÓXICO' if enh_xgb_pred == 1 else 'NO TÓXICO'} (pred={enh_xgb_pred})")
    print(f"  Probabilidad: No Tóxico={enh_xgb_proba[0]:.4f}, Tóxico={enh_xgb_proba[1]:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    if baseline_lr_pred == 1 and enh_lr_pred == 1:
        print("✅ AMBOS pipelines detectan correctamente como TÓXICO (Logistic Regression)")
    elif baseline_lr_pred == 0 and enh_lr_pred == 1:
        print("✨ ENHANCED detecta como TÓXICO, baseline falla (Logistic Regression)")
    elif baseline_lr_pred == 1 and enh_lr_pred == 0:
        print("⚠️  BASELINE detecta como TÓXICO, enhanced falla (Logistic Regression)")
    else:
        print("❌ AMBOS pipelines fallan (Logistic Regression)")

    if baseline_xgb_pred == 1 and enh_xgb_pred == 1:
        print("✅ AMBOS pipelines detectan correctamente como TÓXICO (XGBoost)")
    elif baseline_xgb_pred == 0 and enh_xgb_pred == 1:
        print("✨ ENHANCED detecta como TÓXICO, baseline falla (XGBoost)")
    elif baseline_xgb_pred == 1 and enh_xgb_pred == 0:
        print("⚠️  BASELINE detecta como TÓXICO, enhanced falla (XGBoost)")
    else:
        print("❌ AMBOS pipelines fallan (XGBoost)")

    print("\nFeatures adicionales del texto:")
    print(f"  - Longitud: {df_test['text_length'].values[0]} caracteres")
    print(f"  - Palabras: {df_test['word_count'].values[0]}")
    print(f"  - Tiene mayúsculas: {df_test['has_uppercase'].values[0]}")
    print(f"  - Ratio ALL CAPS: {df_test['all_caps_ratio'].values[0]:.2f}")
    print(f"  - Exclamaciones: {df_test['exclamation_count'].values[0]}")
    print(f"  - Preguntas: {df_test['question_count'].values[0]}")


if __name__ == "__main__":
    compare_pipelines()
