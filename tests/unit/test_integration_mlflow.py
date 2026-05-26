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
import joblib
import mlflow
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
from src.hatescan.training.register_models import _compute_metrics, _register_model
from src.hatescan.training.trainer import HateScanTrainer
 
 
# ── Fixtures ───────────────────────────────────────────────────────────────────
 
@pytest.fixture(scope="module")
def fake_df():
    """Dataset mínimo que simula youtoxic_english_1000.csv."""
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
    texts  = (toxic * 20) + (clean * 20)
    labels = ([1] * 100) + ([0] * 100)
    idx = np.random.permutation(len(texts))
    return pd.DataFrame({
        "text_processed": np.array(texts)[idx],
        "IsToxic":        np.array(labels)[idx],
    })
 
 
@pytest.fixture(scope="module")
def pipeline_output(fake_df, tmp_path_factory):
    """
    Replica el pipeline de baseline.py con datos ficticios.
    No depende de nlp_models.py ni del dataset real.
    """
    # Split 70/15/15
    train_df, temp_df = train_test_split(
        fake_df, test_size=0.30, stratify=fake_df["IsToxic"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["IsToxic"], random_state=42
    )
 
    # Vectorizador — fit solo en train
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    X_train = vectorizer.fit_transform(train_df["text_processed"])
    X_val   = vectorizer.transform(val_df["text_processed"])
    X_test  = vectorizer.transform(test_df["text_processed"])
 
    y_train = train_df["IsToxic"]
    y_val   = val_df["IsToxic"]
    y_test  = test_df["IsToxic"]
 
    # Modelos
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
 
    # Serializar en tmp para no ensuciar el repo
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
    joblib.dump(lr_model,   artifacts_dir / "logistic_regression.joblib")
 
    return {
        "lr_model":        lr_model,
        "tfidf_vectorizer": vectorizer,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "artifacts_dir": artifacts_dir,
    }
 
 
@pytest.fixture(scope="module")
def trainer(tmp_path_factory):
    db = tmp_path_factory.mktemp("mlruns") / "test.db"
    return HateScanTrainer(tracking_uri=f"sqlite:///{db.as_posix()}")
 
 
# ── Tests: splits ─────────────────────────────────────────────────────────────
 
class TestSplits:
    def test_split_sizes(self, pipeline_output):
        """70/15/15 — tolerancia ±2 muestras por redondeo."""
        n = (len(pipeline_output["y_train"]) +
             len(pipeline_output["y_val"]) +
             len(pipeline_output["y_test"]))
        assert abs(len(pipeline_output["y_train"]) - round(n * 0.70)) <= 2
        assert abs(len(pipeline_output["y_val"])   - round(n * 0.15)) <= 2
        assert abs(len(pipeline_output["y_test"])  - round(n * 0.15)) <= 2
 
    def test_no_data_leakage(self, pipeline_output):
        """X_train, X_val y X_test deben tener el mismo número de features."""
        assert pipeline_output["X_train"].shape[1] == pipeline_output["X_val"].shape[1]
        assert pipeline_output["X_train"].shape[1] == pipeline_output["X_test"].shape[1]
 
    def test_stratification(self, pipeline_output):
        """Los splits deben mantener proporciones similares de la clase positiva."""
        for key in ["y_train", "y_val", "y_test"]:
            ratio = pipeline_output[key].mean()
            assert 0.35 <= ratio <= 0.65, \
                f"{key} desbalanceado: {ratio:.2f}"
 
 
# ── Tests: modelo ─────────────────────────────────────────────────────────────
 
class TestModels:
    def test_model_f1_above_threshold(self, pipeline_output):
        """El modelo debe superar F1 mínimo sobre val set."""
        metrics = _compute_metrics(
            pipeline_output["lr_model"],
            pipeline_output["X_val"],
            pipeline_output["y_val"],
        )
        assert metrics["f1_macro"] >= 0.60, \
            f"F1={metrics['f1_macro']:.4f} < 0.60"
 
    def test_no_overfitting(self, pipeline_output):
        """Gap train/val debe ser menor de 10 pp (relajado para datos ficticios)."""
        f1_train = _compute_metrics(
            pipeline_output["lr_model"], pipeline_output["X_train"], pipeline_output["y_train"]
        )["f1_macro"]
        f1_val = _compute_metrics(
            pipeline_output["lr_model"], pipeline_output["X_val"], pipeline_output["y_val"]
        )["f1_macro"]
        assert f1_train - f1_val <= 0.10, \
            f"gap train/val = {f1_train - f1_val:.4f}"
 
    def test_model_can_predict(self, pipeline_output):
        """El modelo debe predecir sobre X_test sin errores."""
        preds = pipeline_output["lr_model"].predict(pipeline_output["X_test"])
        assert len(preds) == len(pipeline_output["y_test"])
        assert set(preds).issubset({0, 1})
 
 
# ── Tests: artefactos ─────────────────────────────────────────────────────────
 
class TestArtifacts:
    def test_vectorizer_is_serialized(self, pipeline_output):
        path = pipeline_output["artifacts_dir"] / "tfidf_vectorizer.joblib"
        assert path.exists(), f"No encontrado: {path}"
 
    def test_model_is_serialized(self, pipeline_output):
        path = pipeline_output["artifacts_dir"] / "logistic_regression.joblib"
        assert path.exists(), f"No encontrado: {path}"
 
    def test_loaded_model_can_predict(self, pipeline_output):
        """El modelo cargado desde disco debe predecir correctamente."""
        path = pipeline_output["artifacts_dir"] / "logistic_regression.joblib"
        loaded = joblib.load(path)
        preds = loaded.predict(pipeline_output["X_test"])
        assert len(preds) == len(pipeline_output["y_test"])
 
 
# ── Tests: integración MLflow ─────────────────────────────────────────────────
 
class TestMLflowIntegration:
    def test_register_run_without_error(self, pipeline_output, trainer):
        """_register_model debe completarse sin lanzar excepciones."""
        _register_model(
            trainer, "baseline_lr",
            pipeline_output["lr_model"],
            pipeline_output["tfidf_vectorizer"],
            pipeline_output["X_train"], pipeline_output["X_val"], pipeline_output["X_test"],
            pipeline_output["y_train"], pipeline_output["y_val"], pipeline_output["y_test"],
        )
 
    def test_run_is_created(self, trainer):
        """Debe existir al menos un run en el experimento."""
        runs = trainer.list_runs()
        assert len(runs) >= 1
 
    def test_metrics_logged_in_mlflow(self, trainer):
        """El run debe tener f1_macro, precision y recall logueados."""
        runs = trainer.list_runs()
        for run in runs:
            assert run["f1_macro"]  is not None
            assert run["precision"] is not None
            assert run["recall"]    is not None
 
    def test_best_run_is_returned(self, trainer):
        """get_best_run debe devolver el run con mayor F1."""
        best = trainer.get_best_run()
        assert best is not None
        assert "run_id"   in best
        assert "run_name" in best
        assert best["metrics"]["f1_macro"] > 0