"""Unit tests for src/preprocess.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    drop_high_missing,
    drop_zero_variance,
    impute_missing,
    remove_low_variance,
    run_preprocessing_pipeline,
    scale_features,
)


@pytest.fixture
def sample_df():
    """Small frame with mixed missingness, a constant column, and outliers."""
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "good": rng.normal(size=n),
            "constant": np.zeros(n),
            "mostly_missing": [np.nan] * 45 + list(rng.normal(size=5)),
            "some_missing": [np.nan if i % 5 == 0 else float(i) for i in range(n)],
        }
    )


def test_drop_high_missing_drops_above_threshold(sample_df):
    out, dropped = drop_high_missing(sample_df, threshold=0.5)
    assert "mostly_missing" in dropped
    assert "mostly_missing" not in out.columns
    assert "good" in out.columns


def test_drop_high_missing_returns_tuple_type(sample_df):
    result = drop_high_missing(sample_df, threshold=0.5)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], list)


def test_drop_zero_variance_removes_constant_column(sample_df):
    filled = sample_df.fillna(0)
    out, dropped = drop_zero_variance(filled)
    assert "constant" in dropped
    assert "constant" not in out.columns


def test_impute_missing_fills_all_nans(sample_df):
    out, imputer = impute_missing(sample_df, strategy="median", fit=True)
    assert out.isnull().sum().sum() == 0
    assert imputer is not None


def test_impute_missing_inference_requires_fitted_imputer(sample_df):
    with pytest.raises(ValueError):
        impute_missing(sample_df, fit=False, imputer=None)


def test_impute_missing_inference_reuses_imputer(sample_df):
    out_train, imputer = impute_missing(sample_df, fit=True)
    new = sample_df.copy()
    new.iloc[0, 0] = np.nan
    out_inf, _ = impute_missing(new, fit=False, imputer=imputer)
    assert out_inf.isnull().sum().sum() == 0


def test_remove_low_variance_drops_constant(sample_df):
    filled = sample_df.fillna(0)
    out, selector = remove_low_variance(filled, threshold=0.0, fit=True)
    assert "constant" not in out.columns
    assert selector is not None


def test_scale_features_centers_and_scales(sample_df):
    filled = sample_df.fillna(0)
    out, scaler = scale_features(filled, fit=True)
    means = out.mean().abs()
    assert (means < 1e-9).all()
    assert scaler is not None


def test_run_preprocessing_pipeline_round_trip(sample_df):
    """fit=True artifacts should reproduce identical output in fit=False mode."""
    filled = sample_df.fillna(0)
    out_train, artifacts = run_preprocessing_pipeline(filled, variance_threshold=0.0)
    out_inf, _ = run_preprocessing_pipeline(filled, fit=False, artifacts=artifacts)
    pd.testing.assert_frame_equal(out_train, out_inf)


def test_run_preprocessing_pipeline_inference_requires_artifacts(sample_df):
    with pytest.raises(ValueError):
        run_preprocessing_pipeline(sample_df.fillna(0), fit=False, artifacts=None)
