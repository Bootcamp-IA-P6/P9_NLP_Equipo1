import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.hatescan.preprocessing import TextTokenizer

DEFAULT_ARTIFACT_DIR = os.path.join("models", "artifacts", "baseline")
DATA_PATH = os.path.join("data", "raw", "youtoxic_english_1000.csv")


@dataclass
class ModelMetrics:
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc: float
    confusion_matrix: list[list[int]]


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_text(data_frame: pd.DataFrame) -> pd.DataFrame:
    tokenizer = TextTokenizer()
    data_frame = data_frame.copy()
    data_frame["text_processed"] = data_frame["Text"].fillna("").apply(tokenizer.process)
    return data_frame


def split_dataset(
    data_frame: pd.DataFrame,
    target_column: str = "IsToxic",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        data_frame,
        test_size=0.30,
        stratify=data_frame[target_column],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[target_column],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def build_vectorizer(
    corpus: pd.Series,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 10000,
) -> tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def transform_corpus(vectorizer: TfidfVectorizer, corpus: pd.Series) -> any:
    return vectorizer.transform(corpus)


def train_logistic_regression(X, y) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model


def train_xgboost(X, y, scale_pos_weight: float) -> XGBClassifier:
    model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_random_forest(X, y) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def compute_class_weights(y) -> float:
    negatives = (y == 0).sum()
    positives = (y == 1).sum()
    return float(negatives / positives) if positives > 0 else 1.0


def evaluate_model(model, X, y) -> ModelMetrics:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    return ModelMetrics(
        f1_macro=f1_score(y, predictions, average="macro"),
        precision_macro=precision_score(y, predictions, average="macro", zero_division=0),
        recall_macro=recall_score(y, predictions, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y, probabilities),
        confusion_matrix=confusion_matrix(y, predictions).tolist(),
    )


def cross_validate_model(model, X, y, n_splits: int = 5) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"f1_macro": [], "precision_macro": [], "recall_macro": [], "roc_auc": []}

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        predictions = model.predict(X_valid)

        metrics["f1_macro"].append(f1_score(y_valid, predictions, average="macro"))
        metrics["precision_macro"].append(precision_score(y_valid, predictions, average="macro", zero_division=0))
        metrics["recall_macro"].append(recall_score(y_valid, predictions, average="macro", zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_valid, probabilities))

    return {k: float(sum(v) / len(v)) for k, v in metrics.items()}


def save_artifact(object_to_save, filename: str) -> str:
    os.makedirs(DEFAULT_ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(DEFAULT_ARTIFACT_DIR, filename)
    joblib.dump(object_to_save, path)
    return path


def run_training(include_random_forest: bool = False) -> dict[str, ModelMetrics]:
    df = load_dataset()
    df = preprocess_text(df)

    train_df, val_df, test_df = split_dataset(df)
    vectorizer, X_train = build_vectorizer(train_df["text_processed"])
    X_val = transform_corpus(vectorizer, val_df["text_processed"])
    X_test = transform_corpus(vectorizer, test_df["text_processed"])
    y_train, y_val, y_test = train_df["IsToxic"], val_df["IsToxic"], test_df["IsToxic"]

    scale_pos_weight = compute_class_weights(y_train)

    lr_model = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, scale_pos_weight)
    rf_model = None
    if include_random_forest:
        rf_model = train_random_forest(X_train, y_train)

    save_artifact(vectorizer, "tfidf_vectorizer.joblib")
    save_artifact(lr_model, "logistic_regression.joblib")
    save_artifact(xgb_model, "xgboost_classifier.joblib")
    if rf_model is not None:
        save_artifact(rf_model, "random_forest.joblib")

    results = {
        "logistic_regression": evaluate_model(lr_model, X_test, y_test),
        "xgboost": evaluate_model(xgb_model, X_test, y_test),
    }
    if rf_model is not None:
        results["random_forest"] = evaluate_model(rf_model, X_test, y_test)

    return results


def print_metrics(metrics: ModelMetrics) -> None:
    print(f"F1-macro: {metrics.f1_macro:.4f}")
    print(f"Precision-macro: {metrics.precision_macro:.4f}")
    print(f"Recall-macro: {metrics.recall_macro:.4f}")
    print(f"ROC-AUC: {metrics.roc_auc:.4f}")
    print("Confusion matrix:")
    print(metrics.confusion_matrix)
