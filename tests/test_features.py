"""Unit tests for src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_balanced_weights,
    drop_highly_correlated,
    get_feature_stats,
    get_imbalance_summary,
    select_top_k_by_mutual_info,
)


@pytest.fixture
def correlated_df():
    """Two highly correlated cols + one independent col."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    return pd.DataFrame({
        "a": base,
        "a_dup": base + rng.normal(scale=1e-4, size=200),
        "b": rng.normal(size=200),
    })


@pytest.fixture
def imbalanced_y():
    return pd.Series([0] * 90 + [1] * 10)


def test_drop_highly_correlated_removes_one_of_each_pair(correlated_df):
    out, dropped = drop_highly_correlated(correlated_df, threshold=0.95)
    assert len(dropped) == 1
    assert dropped[0] in {"a", "a_dup"}
    assert "b" in out.columns
    assert out.shape[1] == 2


def test_drop_highly_correlated_keeps_uncorrelated_features():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x": rng.normal(size=100),
        "y": rng.normal(size=100),
        "z": rng.normal(size=100),
    })
    out, dropped = drop_highly_correlated(df, threshold=0.95)
    assert dropped == []
    assert out.shape == df.shape


def test_select_top_k_by_mutual_info_returns_correct_shape(correlated_df, imbalanced_y):
    y = pd.Series([0, 1] * 100)
    out, scores = select_top_k_by_mutual_info(correlated_df, y, k=2, random_state=0)
    assert out.shape[1] == 2
    assert len(scores) == correlated_df.shape[1]
    assert scores.is_monotonic_decreasing


def test_compute_balanced_weights_inverse_frequency(imbalanced_y):
    weights = compute_balanced_weights(imbalanced_y)
    assert weights[1] > weights[0]
    assert pytest.approx(weights[1] / weights[0], rel=0.1) == 90 / 10


def test_get_feature_stats_columns(correlated_df, imbalanced_y):
    y = pd.Series([0, 1] * 100)
    stats = get_feature_stats(correlated_df, y)
    for col in ["mean", "std", "min", "max", "missing_pct", "corr_with_target", "abs_corr_with_target"]:
        assert col in stats.columns


def test_get_imbalance_summary_pct_sums_to_100(imbalanced_y):
    summary = get_imbalance_summary(imbalanced_y)
    assert pytest.approx(summary["pct"].sum(), rel=1e-6) == 100.0
