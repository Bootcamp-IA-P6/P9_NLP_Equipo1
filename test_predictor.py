"""
HateScan · test_predictor.py  (raíz del proyecto)

Prueba el predictor sin Streamlit ni FastAPI.
Muestra el JSON de salida en consola y permite activar/desactivar
la persistencia en Supabase.

Uso:
    # Solo inferencia, sin guardar en Supabase
    uv run python test_predictor.py

    # Con persistencia en Supabase
    uv run python test_predictor.py --save-db

    # Cambiar umbral
    uv run python test_predictor.py --threshold 0.40
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hatescan.models.predictor import HateScanPredictor

# ── Comentarios de prueba ─────────────────────────────────────────────────────
# Mezcla de comentarios claramente tóxicos, borderline y neutros
# para verificar que el modelo responde como esperamos.

run_id = uuid.uuid4().hex[:6]

TEST_COMMENTS = [
    {
        "comment_id": f"test_{run_id}_001",
        "text": "you are so stupid and pathetic, go kill yourself",
    },
    {
        "comment_id": f"test_{run_id}_002",
        "text": "great video, loved it! very informative thanks",
    },
    {
        "comment_id": f"test_{run_id}_003",
        "text": "all people from that country are criminals and should be deported",
    },
    {
        "comment_id": f"test_{run_id}_004",
        "text": "I disagree with your opinion but I respect your perspective",
    },
    {
        "comment_id": f"test_{run_id}_005",
        "text": "this is pure propaganda, wake up sheeple!!!",
    },
    {
        "comment_id": f"test_{run_id}_006",
        "text": "amazing content as always, keep it up!",
    },
    {
        "comment_id": f"test_{run_id}_007",
        "text": "dirty immigrants stealing our jobs, go back to your country",
    },
    {
        "comment_id": f"test_{run_id}_008",
        "text": "the algorithm on this platform is completely broken",
    },
]


def print_summary(result: dict) -> None:
    """Imprime un resumen legible además del JSON completo."""
    print("\n" + "═" * 60)
    print("  RESUMEN")
    print("═" * 60)
    print(f"  Video URL     : {result['video_url']}")
    print(f"  Modelo        : {result['model_used']}")
    print(f"  Total         : {result['total_comments']} comentarios")
    print(f"  Tóxicos       : {result['toxic_count']}")
    print(f"  No tóxicos    : {result['non_toxic_count']}")
    print("═" * 60)

    print("\n  DETALLE POR COMENTARIO")
    print("─" * 60)
    for c in result["comments"]:
        toxic_icon = "🔴" if c["is_toxic"] else "🟢"
        
        # Leemos de categories para armar los tags
        cats = c.get("categories", {})
        flags = []
        if cats.get("is_hatespeech"): flags.append("hatespeech")
        if cats.get("is_racist"):     flags.append("racist")
        if cats.get("is_obscene"):    flags.append("obscene")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"  {toxic_icon} [{c['comment_id']}] conf={c['confidence']:.3f}{flag_str}")
        print(f"     \"{c['text_original'][:70]}\"")
    print("─" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prueba el HateScanPredictor")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Umbral de clasificación (default: 0.45)")
    p.add_argument("--save-db",   action="store_true",
                   help="Persistir resultados en Supabase")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\nCargando HateScanPredictor (threshold={args.threshold})...")
    predictor = HateScanPredictor(threshold=args.threshold)

    result = predictor.predict(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        video_id="dQw4w9WgXcQ",
        video_title="[TEST] Never Gonna Give You Up",
        comments=TEST_COMMENTS,
        user_session="test_user@hatescan.dev",
        save_to_db=args.save_db,
    )

    print_summary(result)

    print("\n  JSON COMPLETO")
    print("─" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not args.save_db:
        print("\n  💡 Para guardar en Supabase: uv run python test_predictor.py --save-db")
