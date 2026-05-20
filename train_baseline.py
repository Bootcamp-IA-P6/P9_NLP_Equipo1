from src.hatescan.training import run_training


if __name__ == "__main__":
    results = run_training(include_random_forest=False)
    print("\n=== EVALUACIÓN FINAL EN TEST SET ===")
    for model_name, metrics in results.items():
        print(f"\nModelo: {model_name}")
        print("-" * 40)
        print(f"F1-macro: {metrics.f1_macro:.4f}")
        print(f"Precision-macro: {metrics.precision_macro:.4f}")
        print(f"Recall-macro: {metrics.recall_macro:.4f}")
        print(f"ROC-AUC: {metrics.roc_auc:.4f}")
        print("Confusion matrix:")
        print(metrics.confusion_matrix)
