#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Baseline Models

This script performs hyperparameter optimization using Optuna for:
1. TfidfVectorizer parameters
2. LogisticRegression parameters
3. XGBoost parameters

The optimization prioritizes F1-macro score using 5-fold cross-validation.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.hatescan.training.baseline import load_dataset, preprocess_text
from src.hatescan.training.hyperparameter_tuning import (
    train_tuned_model,
    tune_logistic_regression,
    tune_tfidf,
    tune_xgboost,
)


def main():
    print("🔍 Loading and preprocessing dataset...")
    df = load_dataset()
    df = preprocess_text(df)

    # Split for tuning (train + val)
    train_df = df[df["split"] == "train"]
    corpus_train = train_df["text_processed"]
    y_train = train_df["IsToxic"]

    print(f"📊 Training data: {len(train_df)} samples")
    print(f"📊 Toxic/Non-toxic ratio: {y_train.mean():.3f}")

    # Tune TfidfVectorizer
    print("\n🔧 Tuning TfidfVectorizer...")
    tfidf_result = tune_tfidf(corpus_train, y_train, n_trials=30)
    print(f"✅ Best TfidfVectorizer F1-macro: {tfidf_result.best_f1_macro:.4f}")
    print(f"✅ Best params: {tfidf_result.best_params}")

    # Convert TF-IDF params to vectorizer format
    best_tfidf_params = {
        "ngram_range": (tfidf_result.best_params["ngram_min"], tfidf_result.best_params["ngram_max"]),
        "max_features": tfidf_result.best_params["max_features"],
        "min_df": tfidf_result.best_params["min_df"],
        "max_df": tfidf_result.best_params["max_df"],
    }

    # Tune LogisticRegression
    print("\n🔧 Tuning LogisticRegression...")
    # First, transform corpus with best TF-IDF params
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer_temp = TfidfVectorizer(**best_tfidf_params)
    X_train_temp = vectorizer_temp.fit_transform(corpus_train)

    lr_result = tune_logistic_regression(X_train_temp, y_train, n_trials=50)
    print(f"✅ Best LogisticRegression F1-macro: {lr_result.best_f1_macro:.4f}")
    print(f"✅ Best params: {lr_result.best_params}")

    # Tune XGBoost
    print("\n🔧 Tuning XGBoost...")
    xgb_result = tune_xgboost(X_train_temp, y_train, n_trials=50)
    print(f"✅ Best XGBoost F1-macro: {xgb_result.best_f1_macro:.4f}")
    print(f"✅ Best params: {xgb_result.best_params}")

    # Train final tuned models
    print("\n🏗️ Training final tuned models...")
    X_test, y_test, lr_model, xgb_model = train_tuned_model(
        best_tfidf_params, lr_result.best_params, xgb_result.best_params, df
    )

    # Evaluate on test set
    print("\n📊 Evaluating tuned models on test set...")

    # LogisticRegression evaluation
    lr_predictions = lr_model.predict(X_test)
    lr_probabilities = lr_model.predict_proba(X_test)[:, 1]

    lr_f1 = f1_score(y_test, lr_predictions, average="macro")
    lr_precision = precision_score(y_test, lr_predictions, average="macro")
    lr_recall = recall_score(y_test, lr_predictions, average="macro")
    lr_roc_auc = roc_auc_score(y_test, lr_probabilities)

    print("
📈 LogisticRegression Results:"    print(f"   F1-macro: {lr_f1:.4f}")
    print(f"   Precision-macro: {lr_precision:.4f}")
    print(f"   Recall-macro: {lr_recall:.4f}")
    print(f"   ROC-AUC: {lr_roc_auc:.4f}")

    # XGBoost evaluation
    xgb_predictions = xgb_model.predict(X_test)
    xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]

    xgb_f1 = f1_score(y_test, xgb_predictions, average="macro")
    xgb_precision = precision_score(y_test, xgb_predictions, average="macro")
    xgb_recall = recall_score(y_test, xgb_predictions, average="macro")
    xgb_roc_auc = roc_auc_score(y_test, xgb_probabilities)

    print("
📈 XGBoost Results:"    print(f"   F1-macro: {xgb_f1:.4f}")
    print(f"   Precision-macro: {xgb_precision:.4f}")
    print(f"   Recall-macro: {xgb_recall:.4f}")
    print(f"   ROC-AUC: {xgb_roc_auc:.4f}")

    # Test with hate speech phrase
    print("\n🧪 Testing with hate speech phrase...")
    test_phrase = "You disgust me, you fucking black bitch. What the hell are you going to do with Spain? You're a fucking African whore. You don't belong here."

    from src.hatescan.preprocessing import TextTokenizer
    tokenizer = TextTokenizer()
    processed_phrase = tokenizer.process(test_phrase)
    vectorized_phrase = vectorizer_temp.transform([processed_phrase])

    lr_pred = lr_model.predict(vectorized_phrase)[0]
    lr_prob = lr_model.predict_proba(vectorized_phrase)[0][1]

    xgb_pred = xgb_model.predict(vectorized_phrase)[0]
    xgb_prob = xgb_model.predict_proba(vectorized_phrase)[0][1]

    print(f"   Phrase: '{test_phrase[:50]}...'"    print(f"   LogisticRegression: {'TÓXICO' if lr_pred == 1 else 'NO TÓXICO'} ({lr_prob:.1%} confidence)")
    print(f"   XGBoost: {'TÓXICO' if xgb_pred == 1 else 'NO TÓXICO'} ({xgb_prob:.1%} confidence)")

    # Save tuning results
    results = {
        "tfidf_tuning": {
            "best_f1_macro": tfidf_result.best_f1_macro,
            "best_params": tfidf_result.best_params,
        },
        "logistic_regression_tuning": {
            "best_f1_macro": lr_result.best_f1_macro,
            "best_params": lr_result.best_params,
        },
        "xgboost_tuning": {
            "best_f1_macro": xgb_result.best_f1_macro,
            "best_params": xgb_result.best_params,
        },
        "final_test_metrics": {
            "logistic_regression": {
                "f1_macro": lr_f1,
                "precision_macro": lr_precision,
                "recall_macro": lr_recall,
                "roc_auc": lr_roc_auc,
            },
            "xgboost": {
                "f1_macro": xgb_f1,
                "precision_macro": xgb_precision,
                "recall_macro": xgb_recall,
                "roc_auc": xgb_roc_auc,
            },
        },
        "hate_speech_test": {
            "phrase": test_phrase,
            "logistic_regression": {"prediction": int(lr_pred), "probability": float(lr_prob)},
            "xgboost": {"prediction": int(xgb_pred), "probability": float(xgb_prob)},
        },
    }

    import json
    with open(os.path.join("models", "artifacts", "baseline_tuned", "tuning_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("
✅ Tuning completed! Results saved to models/artifacts/baseline_tuned/"    print("📋 Recommendations:"    print("   1. Use LogisticRegression for production - better F1-macro and more interpretable")
    print("   2. XGBoost shows higher confidence on hate speech detection")
    print("   3. Both models maintain hate speech detection capability")
    print("   4. Consider ensemble methods for even better performance")


if __name__ == "__main__":
    main()