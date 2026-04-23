"""Unit tests for src/models.py."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models import (
    build_random_forest,
    cross_validated_metrics,
    find_best_threshold,
    load_threshold,
    save_threshold,
)


@pytest.fixture
def imbalanced_dataset():
    """Synthetic 90/10 binary classification frame."""
    X, y = make_classification(
        n_samples=400,
        n_features=20,
        n_informative=8,
        weights=[0.9, 0.1],
        random_state=0,
    )
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]),
        pd.Series(y),
    )


def test_build_random_forest_uses_class_balanced():
    rf = build_random_forest(random_state=0)
    assert rf.class_weight == "balanced"
    assert rf.random_state == 0


def test_cross_validated_metrics_stable_on_fixed_seed(imbalanced_dataset):
    X, y = imbalanced_dataset
    cv = cross_validated_metrics(
        lambda: build_random_forest(random_state=0),
        X, y, n_splits=3, random_state=0,
    )
    assert cv.n_splits == 3
    assert 0.0 <= cv.roc_auc_mean <= 1.0
    assert 0.0 <= cv.pr_auc_mean <= 1.0
    assert 0.0 <= cv.fail_f1_mean <= 1.0
    assert cv.roc_auc_std >= 0.0
    assert "roc_auc_mean" in cv.as_dict()


def test_find_best_threshold_returns_valid_threshold(imbalanced_dataset):
    X, y = imbalanced_dataset
    rf = build_random_forest(random_state=0).fit(X, y)
    prob = rf.predict_proba(X)[:, 1]

    threshold, diag = find_best_threshold(y, prob, objective="f1")
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= diag["f1"] <= 1.0
    assert diag["objective"] == "f1"


def test_find_best_threshold_min_recall_floor(imbalanced_dataset):
    X, y = imbalanced_dataset
    rf = build_random_forest(random_state=0).fit(X, y)
    prob = rf.predict_proba(X)[:, 1]

    threshold, diag = find_best_threshold(y, prob, objective="f1", min_recall=0.8)
    assert diag["recall"] >= 0.8 - 1e-9


def test_find_best_threshold_rejects_unknown_objective():
    y = pd.Series([0, 1, 0, 1])
    prob = np.array([0.1, 0.9, 0.4, 0.6])
    with pytest.raises(ValueError):
        find_best_threshold(y, prob, objective="not-a-thing")


def test_save_and_load_threshold_roundtrip(tmp_path):
    save_threshold(0.37, {"precision": 0.5, "recall": 0.6, "f1": 0.55}, str(tmp_path))
    assert (tmp_path / "threshold.json").exists()

    threshold = load_threshold(str(tmp_path))
    assert pytest.approx(threshold) == 0.37

    with open(tmp_path / "threshold.json") as f:
        body = json.load(f)
    assert body["threshold"] == 0.37
    assert body["precision"] == 0.5


def test_load_threshold_falls_back_to_default(tmp_path):
    assert load_threshold(str(tmp_path), default=0.42) == 0.42
