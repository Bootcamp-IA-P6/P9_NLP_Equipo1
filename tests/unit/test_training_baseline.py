import pandas as pd

from src.hatescan.training.baseline import (
    build_vectorizer,
    compute_class_weights,
    split_dataset,
    train_logistic_regression,
    transform_corpus,
)


def test_split_dataset_preserves_stratification():
    df = pd.DataFrame({
        "Text": [f"text_{i}" for i in range(20)],
        "IsToxic": [0] * 10 + [1] * 10,
    })
    train_df, val_df, test_df = split_dataset(df)

    assert len(train_df) == 14
    assert len(val_df) == 3
    assert len(test_df) == 3
    assert train_df["IsToxic"].mean() == df["IsToxic"].mean()


def test_vectorizer_builds_feature_matrix():
    corpus = pd.Series(["hello world", "hate speech", "this is toxic"])
    vectorizer, matrix = build_vectorizer(corpus, max_features=10)

    assert matrix.shape[0] == 3
    assert matrix.shape[1] <= 10
    assert "hate" in vectorizer.get_feature_names_out()


def test_compute_class_weights():
    weights = compute_class_weights(pd.Series([0, 0, 0, 1]))
    assert weights == 3.0


def test_train_logistic_regression_runs():
    corpus = pd.Series(["hate speech", "good comment", "bad words", "nice video"])
    vectorizer, X = build_vectorizer(corpus, max_features=10)
    y = pd.Series([1, 0, 1, 0])

    model = train_logistic_regression(X, y)
    predictions = model.predict(X)

    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})


def test_transform_corpus_works():
    corpus = pd.Series(["hello world", "hate speech"])
    vectorizer, _ = build_vectorizer(corpus, max_features=10)
    transformed = transform_corpus(vectorizer, pd.Series(["hello world"]))

    assert transformed.shape[0] == 1
    assert transformed.shape[1] <= 10
