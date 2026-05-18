import os
from dataclasses import dataclass

import joblib
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import Trial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.hatescan.preprocessing import TextTokenizer
from src.hatescan.training.baseline import (
    load_dataset,
    preprocess_text,
    split_dataset,
)

DEFAULT_ARTIFACT_DIR = os.path.join("models", "artifacts", "baseline")
TUNING_ARTIFACT_DIR = os.path.join("models", "artifacts", "baseline_tuned")


@dataclass
class TuningResult:
    best_trial_number: int
    best_f1_macro: float
    best_params: dict
    model_name: str


def objective_logistic_regression(trial: Trial, X_train, y_train) -> float:
    """Objective function for LogisticRegression tuning."""
    C = trial.suggest_float("C", 0.001, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    # Validate solver-penalty combination
    if solver == "lbfgs" and penalty == "l1":
        return 0.0  # Skip invalid combination

    model = LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_fold_train = X_train[train_idx]
        X_fold_valid = X_train[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_valid)
        f1 = f1_score(y_fold_valid, predictions, average="macro", zero_division=0)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def objective_xgboost(trial: Trial, X_train, y_train) -> float:
    """Objective function for XGBoost tuning."""
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)

    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0

    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_fold_train = X_train[train_idx]
        X_fold_valid = X_train[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_valid)
        f1 = f1_score(y_fold_valid, predictions, average="macro", zero_division=0)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def objective_tfidf(trial: Trial, corpus_train, y_train) -> float:
    """Objective function for TfidfVectorizer tuning."""
    ngram_min = trial.suggest_int("ngram_min", 1, 2)
    ngram_max = trial.suggest_int("ngram_max", 1, 3)
    max_features = trial.suggest_int("max_features", 5000, 15000, step=1000)
    min_df = trial.suggest_int("min_df", 1, 5)
    max_df = trial.suggest_float("max_df", 0.7, 1.0)

    if ngram_min > ngram_max:
        return 0.0  # Skip invalid combination

    vectorizer = TfidfVectorizer(
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    X_train = vectorizer.fit_transform(corpus_train)

    model = LogisticRegression(
        C=1.0,
        solver="saga",
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_fold_train = X_train[train_idx]
        X_fold_valid = X_train[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_valid)
        f1 = f1_score(y_fold_valid, predictions, average="macro", zero_division=0)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def tune_logistic_regression(X_train, y_train, n_trials: int = 50) -> TuningResult:
    """Tune LogisticRegression hyperparameters."""
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_logistic_regression(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return TuningResult(
        best_trial_number=study.best_trial.number,
        best_f1_macro=study.best_value,
        best_params=study.best_params,
        model_name="logistic_regression",
    )


def tune_xgboost(X_train, y_train, n_trials: int = 50) -> TuningResult:
    """Tune XGBoost hyperparameters."""
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return TuningResult(
        best_trial_number=study.best_trial.number,
        best_f1_macro=study.best_value,
        best_params=study.best_params,
        model_name="xgboost",
    )


def tune_tfidf(corpus_train, y_train, n_trials: int = 30) -> TuningResult:
    """Tune TfidfVectorizer hyperparameters."""
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_tfidf(trial, corpus_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return TuningResult(
        best_trial_number=study.best_trial.number,
        best_f1_macro=study.best_value,
        best_params=study.best_params,
        model_name="tfidf_vectorizer",
    )


def train_tuned_model(best_params_tfidf, best_params_lr, best_params_xgb, df):
    """Train final models with tuned parameters."""
    train_df, val_df, test_df = split_dataset(df)

    # Build tuned vectorizer
    vectorizer = TfidfVectorizer(**best_params_tfidf)
    X_train = vectorizer.fit_transform(train_df["text_processed"])
    X_val = vectorizer.transform(val_df["text_processed"])
    X_test = vectorizer.transform(test_df["text_processed"])

    y_train, y_val, y_test = train_df["IsToxic"], val_df["IsToxic"], test_df["IsToxic"]

    # Train tuned LogisticRegression
    lr_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        **best_params_lr,
    )
    lr_model.fit(X_train, y_train)

    # Train tuned XGBoost
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0

    xgb_model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        **best_params_xgb,
    )
    xgb_model.fit(X_train, y_train)

    # Save tuned artifacts
    os.makedirs(TUNING_ARTIFACT_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(TUNING_ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(lr_model, os.path.join(TUNING_ARTIFACT_DIR, "logistic_regression.joblib"))
    joblib.dump(xgb_model, os.path.join(TUNING_ARTIFACT_DIR, "xgboost_classifier.joblib"))

    return X_test, y_test, lr_model, xgb_model
