"""
HateScan · run_training_transformer.py  (raíz del proyecto)

Entry point para entrenar el transformer multilabel, paralelo a
run_training_pipeline.py que entrena los modelos clásicos.

Uso rápido:
    uv run python run_training_transformer.py

Con opciones:
    uv run python run_training_transformer.py --epochs 3 --threshold 0.25
    uv run python run_training_transformer.py --no-focal-loss
    uv run python run_training_transformer.py --model cardiffnlp/twitter-roberta-base-offensive

Modelos alternativos probados en hate speech (inglés):
    cardiffnlp/twitter-roberta-base-hate         ← default, mejor para nuestro dominio
    cardiffnlp/twitter-roberta-base-offensive    ← alternativa si hate no converge
    distilbert-base-uncased                      ← más rápido, algo menos preciso
"""

import sys
from pathlib import Path

# Asegura que src/ está en el path igual que el resto de scripts del proyecto
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hatescan.training.train_transformer import parse_args, train, TransformerHyperparams

if __name__ == "__main__":
    args = parse_args()

    if args.eval_only:
        # Solo reevalúa el modelo guardado con el umbral indicado, sin reentrenar
        from hatescan.training.train_transformer import evaluate_saved
        evaluate_saved(threshold=args.threshold)
    else:
        hp = TransformerHyperparams(
            model_name=args.model,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            cls_threshold=args.threshold,
            focal_loss=not args.no_focal_loss,
        )

        metrics = train(hp)

        print("\n── Resultado final ─────────────────────────────────────")
        print(f"  F1-macro  (test): {metrics['f1_macro']:.4f}")
        print(f"  Precision (test): {metrics['precision']:.4f}")
        print(f"  Recall    (test): {metrics['recall']:.4f}")
        print(f"  ROC-AUC   (test): {metrics['roc_auc']:.4f}")
        print("────────────────────────────────────────────────────────")
