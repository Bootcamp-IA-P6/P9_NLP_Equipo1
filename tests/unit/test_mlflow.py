"""
HateScan · ISSUE-06
tests/unit/test_mlflow.py

Tests unitarios del HateScanTrainer:
  - Creación del experimento y reutilización.
  - Logging de parámetros, métricas y artefactos.
  - Validaciones: run_name inválido, métricas faltantes,
    umbral F1 y gap overfitting.
  - Consultas: list_runs y get_best_run.

Ejecutar:
    pytest tests/unit/test_mlflow.py -v
"""

from mlflow import experiments
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
from webcolors import names

from src.hatescan.training.trainer import HateScanTrainer, EXPERIMENT_NAME


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_trainer(tmp_path_factory):
    """Trainer aislado con base de datos temporal."""
    db = tmp_path_factory.mktemp("mlruns") / "test.db"
    uri = f"sqlite:///{db.as_posix()}"   # ← as_posix() corrige las barras en Windows
    return HateScanTrainer(tracking_uri=uri)


@pytest.fixture(scope="module")
def dummy_model_and_vectorizer():
    """Modelo LR mínimo entrenado sobre texto ficticio."""
    corpus = [
        "you are so stupid",
        "i hate you so much",
        "have a nice day",
        "what a lovely morning",
        "kill yourself loser",
        "great job today",
    ]
    labels = [1, 1, 0, 0, 1, 0]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    X = vectorizer.fit_transform(corpus)
    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)
    return model, vectorizer, X, np.array(labels)


@pytest.fixture
def valid_metrics():
    """Métricas sobre val set que superan todos los umbrales."""
    return {
        "f1_macro":  0.78,
        "precision": 0.76,
        "recall":    0.80,
        "roc_auc":   0.85,
    }


@pytest.fixture
def valid_params():
    return {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}


# ── Tests: experimento ─────────────────────────────────────────────────────────

class TestExperimentSetup:
    def test_experiment_is_created(self, tmp_trainer):
        """El experimento 'hatescan_nlp' debe existir tras inicializar el trainer."""
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        assert exp is not None
        assert exp.name == EXPERIMENT_NAME

    def test_second_init_reuses_experiment(self, tmp_trainer):
        """Instanciar el trainer dos veces no duplica el experimento."""
        experiments = mlflow.tracking.MlflowClient().list_experiments()
        names = [e.name for e in experiments]
        assert names.count(EXPERIMENT_NAME) == 1


# ── Tests: log_run ─────────────────────────────────────────────────────────────

class TestLogRun:
    def test_run_is_created_with_status_finished(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, X, y = dummy_model_and_vectorizer
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=valid_metrics,
            X_sample=X, y_sample=y,
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_params_are_logged(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=valid_metrics,
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert run.data.params["C"] == "1.0"
        assert run.data.params["solver"] == "lbfgs"

    def test_required_metrics_are_logged(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=valid_metrics,
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert run.data.metrics["f1_macro"]  == pytest.approx(0.78)
        assert run.data.metrics["precision"] == pytest.approx(0.76)
        assert run.data.metrics["recall"]    == pytest.approx(0.80)
        assert run.data.metrics["roc_auc"]   == pytest.approx(0.85)

    def test_vectorizer_artifact_is_logged(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=valid_metrics,
        )
        artifacts = mlflow.tracking.MlflowClient().list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        assert any("vectorizer" in p for p in artifact_paths), \
            f"Artefacto 'vectorizer' no encontrado. Encontrados: {artifact_paths}"

    def test_train_duration_is_logged(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=valid_metrics,
            train_duration_seconds=3.14,
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert run.data.metrics["train_duration_seconds"] == pytest.approx(3.14)

    def test_f1_train_val_gap_is_logged(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        """Si se pasa f1_train debe registrarse el gap train/val."""
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=valid_metrics,
            f1_train=0.80,  # gap = 0.80 - 0.78 = 0.02 → OK
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert "f1_train_val_gap" in run.data.metrics
        assert run.data.metrics["f1_train_val_gap"] == pytest.approx(0.02, abs=1e-3)

    def test_test_metrics_are_logged_when_provided(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_params
    ):
        """Las métricas _test (informe final) deben registrarse si se incluyen."""
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        metrics_with_test = {
            "f1_macro":       0.78,
            "precision":      0.76,
            "recall":         0.80,
            "roc_auc":        0.85,
            "f1_macro_test":  0.76,
            "precision_test": 0.74,
            "recall_test":    0.78,
            "roc_auc_test":   0.83,
        }
        run_id = tmp_trainer.log_run(
            run_name="baseline_lr",
            model=model, vectorizer=vectorizer,
            params=valid_params, metrics=metrics_with_test,
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert run.data.metrics["f1_macro_test"] == pytest.approx(0.76)


# ── Tests: validaciones ────────────────────────────────────────────────────────

class TestValidations:
    def test_invalid_run_name_raises(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        with pytest.raises(ValueError, match="no válido"):
            tmp_trainer.log_run(
                run_name="modelo_inventado",
                model=model, vectorizer=vectorizer,
                params=valid_params, metrics=valid_metrics,
            )

    def test_missing_required_metric_raises(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        incomplete = {"f1_macro": 0.78, "precision": 0.76}  # faltan recall y roc_auc
        with pytest.raises(ValueError, match="Faltan métricas"):
            tmp_trainer.log_run(
                run_name="baseline_lr",
                model=model, vectorizer=vectorizer,
                params=valid_params, metrics=incomplete,
            )

    def test_f1_below_threshold_raises(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        low_metrics = {"f1_macro": 0.50, "precision": 0.48, "recall": 0.52, "roc_auc": 0.60}
        with pytest.raises(ValueError, match="umbral mínimo"):
            tmp_trainer.log_run(
                run_name="baseline_lr",
                model=model, vectorizer=vectorizer,
                params=valid_params, metrics=low_metrics,
            )

    def test_overfitting_gap_raises(
        self, tmp_trainer, dummy_model_and_vectorizer, valid_metrics, valid_params
    ):
        model, vectorizer, _, _ = dummy_model_and_vectorizer
        with pytest.raises(ValueError, match="overfitting"):
            tmp_trainer.log_run(
                run_name="baseline_lr",
                model=model, vectorizer=vectorizer,
                params=valid_params, metrics=valid_metrics,  # f1_macro val = 0.78
                f1_train=0.95,                               # gap = 0.17 → falla
            )


# ── Tests: consulta ────────────────────────────────────────────────────────────

class TestQueryRuns:
    def test_list_runs_returns_list(self, tmp_trainer):
        result = tmp_trainer.list_runs()
        assert isinstance(result, list)

    def test_list_runs_contain_required_keys(self, tmp_trainer):
        runs = tmp_trainer.list_runs()
        if runs:
            assert "run_id"   in runs[0]
            assert "run_name" in runs[0]
            assert "f1_macro" in runs[0]

    def test_get_best_run_returns_dict_or_none(self, tmp_trainer):
        result = tmp_trainer.get_best_run()
        assert result is None or (isinstance(result, dict) and "run_id" in result)

    def test_get_best_run_has_highest_f1(self, tmp_trainer):
        """El best run debe tener el F1 más alto de todos."""
        best = tmp_trainer.get_best_run()
        if best is None:
            pytest.skip("No hay runs en el experimento")
        all_runs = tmp_trainer.list_runs()
        max_f1 = max(r["f1_macro"] for r in all_runs if r["f1_macro"] is not None)
        assert best["metrics"]["f1_macro"] == pytest.approx(max_f1, abs=1e-4)
