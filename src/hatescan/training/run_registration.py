"""
HateScan · ISSUE-06
src/hatescan/training/run_registration.py

Ejecutar desde la raíz del repo:
    python src/hatescan/training/run_registration.py
"""
import logging
from src.hatescan.training.register_models import register_all

logging.basicConfig(level=logging.INFO, format='%(levelname)s — %(message)s')

if __name__ == "__main__":
    register_all(
        include_random_forest=True,
        include_tuned=True,   # ponlo True si quieres los modelos con Optuna (tarda más)
    )