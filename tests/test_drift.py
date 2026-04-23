"""Unit tests for src/drift.py."""

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    classify_psi,
    compute_drift_from_summary,
    compute_drift_report,
    load_reference_summary,
    population_stability_index,
    save_reference_summary,
)


@pytest.fixture
def reference_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "stable": rng.normal(0, 1, size=500),
        "drifty": rng.normal(0, 1, size=500),
    })


@pytest.fixture
def stable_current(reference_frame):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "stable": rng.normal(0, 1, size=500),
        "drifty": rng.normal(0, 1, size=500),
    })


@pytest.fixture
def drifted_current():
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "stable": rng.normal(0, 1, size=500),
        "drifty": rng.normal(3, 2, size=500),  # mean and variance shift
    })


def test_psi_zero_for_identical_distributions(reference_frame):
    psi = population_stability_index(
        reference_frame["stable"].to_numpy(),
        reference_frame["stable"].to_numpy(),
    )
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_positive_for_shifted_distribution(reference_frame, drifted_current):
    psi = population_stability_index(
        reference_frame["drifty"].to_numpy(),
        drifted_current["drifty"].to_numpy(),
    )
    assert psi > 0.25  # significant drift


def test_psi_handles_nans():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=200)
    cur = np.concatenate([rng.normal(size=190), [np.nan] * 10])
    psi = population_stability_index(ref, cur)
    assert not np.isnan(psi)


def test_classify_psi_bucket_boundaries():
    assert classify_psi(0.05) == "none"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.30) == "significant"
    assert classify_psi(float("nan")) == "unknown"


def test_compute_drift_report_flags_drifted_feature(reference_frame, drifted_current):
    report = compute_drift_report(reference_frame, drifted_current)
    top = report.per_feature.index[0]
    assert top == "drifty"
    assert report.per_feature.loc["drifty", "severity"] == "significant"
    assert report.per_feature.loc["stable", "severity"] in {"none", "moderate"}
    assert report.n_significant >= 1


def test_compute_drift_report_n_counts_match(reference_frame, stable_current):
    report = compute_drift_report(reference_frame, stable_current)
    assert report.n_reference == 500
    assert report.n_current == 500


def test_save_and_load_reference_summary_roundtrip(tmp_path, reference_frame):
    path = tmp_path / "ref.json"
    save_reference_summary(reference_frame, str(path))
    summary = load_reference_summary(str(path))
    assert summary is not None
    assert summary["n_samples"] == 500
    assert set(summary["feature_deciles"].keys()) == {"stable", "drifty"}
    assert len(summary["feature_deciles"]["stable"]) == 11


def test_load_reference_summary_returns_none_when_missing(tmp_path):
    assert load_reference_summary(str(tmp_path / "missing.json")) is None


def test_compute_drift_from_summary_uses_deciles(tmp_path, reference_frame, drifted_current):
    path = tmp_path / "ref.json"
    save_reference_summary(reference_frame, str(path))
    summary = load_reference_summary(str(path))
    report = compute_drift_from_summary(summary, drifted_current)
    assert report.per_feature.loc["drifty", "severity"] == "significant"
    assert report.n_reference == 500
    assert report.n_current == 500
