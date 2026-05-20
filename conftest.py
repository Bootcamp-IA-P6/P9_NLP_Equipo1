"""
conftest.py — raíz del repo.

Añade src/ al path de Python para que todos los tests puedan importar
los módulos del proyecto con:
    from src.hatescan.training.trainer import HateScanTrainer
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
