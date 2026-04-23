"""Unit tests for src/diagnostics.py and src/artifacts.py."""

import os

import pandas as pd
import pytest

from src.artifacts import load_pipeline_artifacts
from src.diagnostics import DiagnosticsPipeline, generate_root_cause_summary

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(REPO_ROOT, "dashboard_assets", "models")
DATA_DIR = os.path.join(REPO_ROOT, "dashboard_assets", "data")


@pytest.fixture(scope="module")
def pipeline():
    p = DiagnosticsPipeline(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    p.load()
    return p


def test_load_pipeline_artifacts_returns_all_pieces():
    artifacts = load_pipeline_artifacts(MODEL_DIR)
    assert artifacts.model is not None
    assert artifacts.var_selector is not None
    assert artifacts.scaler is not None
    assert len(artifacts.input_feature_names) > 0
    assert len(artifacts.corr_kept_cols) > 0
    assert len(artifacts.mi_selected_cols) == 50


def test_pipeline_loads_with_baseline(pipeline):
    assert pipeline.model is not None
    assert pipeline.baseline is not None
    assert pipeline.baseline.n_samples > 0
    assert len(pipeline.baseline.mean) == 50


def test_get_sample_returns_446_features(pipeline):
    sample = pipeline.get_sample(0)
    assert isinstance(sample, pd.Series)
    assert len(sample) == len(pipeline.input_feature_names)


def test_diagnose_by_index_produces_full_report(pipeline):
    diag = pipeline.diagnose_by_index(0, top_k=5)
    assert diag.prediction in {"PASS", "FAIL"}
    assert 0.0 <= diag.probability <= 1.0
    assert len(diag.shap_values) == 50
    assert len(diag.deviations_z) == 50
    assert len(diag.top_shap_features) == 5
    assert len(diag.top_deviated_features) == 5


def test_zscore_sign_matches_direction(pipeline):
    """Positive z-score should mean sample value > baseline mean."""
    diag = pipeline.diagnose_by_index(0, top_k=10)
    for feat in diag.top_shap_features[:5]:
        z = diag.deviations_z[feat]
        sample_val = diag.feature_values[feat]
        baseline_mean = pipeline.baseline.mean[feat]
        if z > 0:
            assert sample_val > baseline_mean
        elif z < 0:
            assert sample_val < baseline_mean


def test_baseline_comparison_df_shape(pipeline):
    diag = pipeline.diagnose_by_index(0, top_k=5)
    df = pipeline.get_baseline_comparison_df(diag, features=diag.top_shap_features[:3])
    assert len(df) == 3
    for col in ["Sensor", "Sample Value", "Baseline Mean", "Baseline Std", "Z-Score", "|SHAP|"]:
        assert col in df.columns


def test_root_cause_summary_includes_prediction(pipeline):
    diag = pipeline.diagnose_by_index(0)
    summary = generate_root_cause_summary(diag, pipeline.baseline)
    assert diag.prediction in summary
    assert "Wafer" in summary


def test_pass_and_fail_indices_are_disjoint(pipeline):
    pass_set = set(pipeline.pass_indices)
    fail_set = set(pipeline.fail_indices)
    assert pass_set.isdisjoint(fail_set)
    assert len(pass_set) + len(fail_set) == pipeline.n_samples
