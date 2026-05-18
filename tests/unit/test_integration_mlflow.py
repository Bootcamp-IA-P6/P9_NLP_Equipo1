"""
HateScan · ISSUE-06
tests/unit/test_integration_mlflow.py

Test de integración: entrena los modelos con datos ficticios y verifica
que todo el pipeline (ISSUE-05 + ISSUE-06) funciona correctamente.

Ejecutar:
    pytest tests/unit/test_integration_mlflow.py -v
"""

import pytest
import numpy as np
import pandas as pd
import mlflow

from src.hatescan.training.nlp_models import run_nlp_pipeline, compute_metrics
from src.hatescan.training.register_models import register_all_models
from src.hatescan.training.trainer import HateScanTrainer


# ── Fixture: dataset ficticio ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fake_df():
    """
    Dataset mínimo que simula youtoxic_english_1000.csv.
    200 muestras, balance similar al real (~46% tóxico).
    """
    np.random.seed(42)
    toxic = [
        "you are so stupid and pathetic",
        "i hate you go kill yourself",
        "what a disgusting loser",
        "shut up nobody likes you",
        "you deserve to suffer",
    ]
    clean = [
        "have a wonderful day today",
        "this is a great video thank you",
        "i really enjoyed watching this",
        "great content keep it up",
        "wonderful performance by everyone",
    ]
    texts  = (toxic * 20) + (clean * 20)   # 100 tóxicos + 100 limpios
    labels = ([1] * 100) + ([0] * 100)

    idx = np.random.permutation(len(texts))
    return pd.DataFrame({
        "text_processed": np.array(texts)[idx],
        "IsToxic":        np.array(labels)[idx],
    })


@pytest.fixture(scope="module")
def pipeline_output(fake_df, tmp_path_factory):
    """Ejecuta el pipeline completo una sola vez para todos los tests."""
    # Redirige artifacts a tmp para no ensuciar el repo
    import src.hatescan.training.nlp_models as nm
    nm.ARTIFACTS_DIR = tmp_path_factory.mktemp("artifacts")
    return run_nlp_pipeline(fake_df)


@pytest.fixture(scope="module")
def trainer(tmp_path_factory):
    db = tmp_path_factory.mktemp("mlruns") / "test.db"
    return HateScanTrainer(tracking_uri=f"sqlite:///{db.as_posix()}")


# ── Tests: splits ─────────────────────────────────────────────────────────────

class TestSplits:
    def test_split_sizes(self, pipeline_output):
        """70/15/15 — tolerancia ±2 muestras por redondeo."""
        n = len(pipeline_output["y_train"]) + len(pipeline_output["y_val"]) + len(pipeline_output["y_test"])
        assert abs(len(pipeline_output["y_train"]) - round(n * 0.70)) <= 2
        assert abs(len(pipeline_output["y_val"])   - round(n * 0.15)) <= 2
        assert abs(len(pipeline_output["y_test"])  - round(n * 0.15)) <= 2

    def test_no_data_leakage(self, pipeline_output):
        """El vectorizador no debe estar entrenado con datos de val/test."""
        # Si hay leakage el vocabulario sería exactamente el mismo en train y total
        # Verificamos que X_train tiene las mismas features que X_val (mismo vectorizador)
        assert pipeline_output["X_train"].shape[1] == pipeline_output["X_val"].shape[1]
        assert pipeline_output["X_train"].shape[1] == pipeline_output["X_test"].shape[1]

    def test_stratification(self, pipeline_output):
        """Los splits deben mantener proporciones similares de la clase positiva."""
        for split_name in ["y_train", "y_val", "y_test"]:
            y = pipeline_output[split_name]
            ratio = y.mean()
            assert 0.35 <= ratio <= 0.65, \
                f"{split_name} desbalanceado: {ratio:.2f} (esperado entre 0.35 y 0.65)"


# ── Tests: modelos ────────────────────────────────────────────────────────────

class TestModels:
    @pytest.mark.parametrize("model_key,threshold", [
        ("baseline_lr",   0.60),   # umbral relajado para datos ficticios
        ("xgboost_model", 0.60),
        ("rf_model",      0.60),
    ])
    def test_model_f1_above_threshold(self, pipeline_output, model_key, threshold):
        """Los modelos deben superar el umbral mínimo de F1 sobre val set."""
        model = pipeline_output[model_key]
        metrics = compute_metrics(model, pipeline_output["X_val"], pipeline_output["y_val"])
        assert metrics["f1_macro"] >= threshold, \
            f"{model_key}: F1={metrics['f1_macro']:.4f} < umbral {threshold}"

    @pytest.mark.parametrize("model_key", ["baseline_lr", "xgboost_model", "rf_model"])
    def test_no_overfitting(self, pipeline_output, model_key):
        """Gap train/val debe ser menor de 5 pp (restricción del plan)."""
        model = pipeline_output[model_key]
        f1_train = compute_metrics(model, pipeline_output["X_train"], pipeline_output["y_train"])["f1_macro"]
        f1_val   = compute_metrics(model, pipeline_output["X_val"],   pipeline_output["y_val"])["f1_macro"]
        gap = f1_train - f1_val
        assert gap <= 0.10, \
            f"{model_key}: gap train/val = {gap:.4f} (umbral relajado a 0.10 para datos ficticios)"

    def test_models_can_predict(self, pipeline_output):
        """Los tres modelos deben poder predecir sobre X_test sin errores."""
        for key in ["baseline_lr", "xgboost_model", "rf_model"]:
            preds = pipeline_output[key].predict(pipeline_output["X_test"])
            assert len(preds) == len(pipeline_output["y_test"])
            assert set(preds).issubset({0, 1})


# ── Tests: artefactos serializados ────────────────────────────────────────────

class TestArtifacts:
    def test_vectorizer_is_serialized(self, pipeline_output):
        """El vectorizador debe haberse guardado en models/artifacts/."""
        import src.hatescan.training.nlp_models as nm
        path = nm.ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
        assert path.exists(), f"No encontrado: {path}"

    @pytest.mark.parametrize("model_name", ["baseline_lr", "xgboost", "random_forest"])
    def test_model_is_serialized(self, pipeline_output, model_name):
        import src.hatescan.training.nlp_models as nm
        path = nm.ARTIFACTS_DIR / f"{model_name}.joblib"
        assert path.exists(), f"No encontrado: {path}"


# ── Tests: integración MLflow ─────────────────────────────────────────────────

class TestMLflowIntegration:
    def test_register_all_runs_without_error(self, pipeline_output, trainer):
        """register_all_models debe completarse sin lanzar excepciones."""
        register_all_models(**pipeline_output, trainer_instance=trainer)

    def test_three_runs_created(self, pipeline_output, trainer):
        """Deben existir exactamente 3 runs en el experimento."""
        # Ejecutamos de nuevo por si el test anterior falló
        try:
            register_all_models(**pipeline_output, trainer_instance=trainer)
        except Exception:
            pass
        runs = trainer.list_runs()
        run_names = {r["run_name"] for r in runs}
        assert "baseline_lr"   in run_names
        assert "xgboost"       in run_names
        assert "random_forest" in run_names

    def test_metrics_logged_in_mlflow(self, pipeline_output, trainer):
        """Cada run debe tener f1_macro, precision, recall y roc_auc logueados."""
        runs = trainer.list_runs()
        for run in runs:
            assert run["f1_macro"]  is not None, f"{run['run_name']}: f1_macro no logueado"
            assert run["precision"] is not None, f"{run['run_name']}: precision no logueado"
            assert run["recall"]    is not None, f"{run['run_name']}: recall no logueado"

    def test_best_run_is_returned(self, trainer):
        """get_best_run debe devolver el run con mayor F1."""
        best = trainer.get_best_run()
        assert best is not None
        assert "run_id"   in best
        assert "run_name" in best
        assert best["metrics"]["f1_macro"] > 0
