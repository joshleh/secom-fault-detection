"""Smoke tests for the SECOM API using FastAPI's TestClient.

Runs against the in-process app (no live server needed) and uses the
committed dashboard_assets/ snapshot so it works in CI without retraining.
"""

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Point the API at the committed dashboard snapshot so tests are self-contained
os.environ.setdefault(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "dashboard_assets", "models"),
)

from api.main import app  # noqa: E402  (import after env var)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features():
    """First wafer from the committed dashboard snapshot (446 features)."""
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "dashboard_assets", "data", "X_clean.csv"
    )
    return pd.read_csv(csv_path).iloc[0].tolist()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    pipe = body["pipeline"]
    assert pipe["input_features"] > 0
    assert pipe["after_corr_filter"] > 0
    assert pipe["after_mi_select"] > 0
    assert pipe["model_trees"] > 0


def test_predict_returns_valid_schema(client, sample_features):
    r = client.post("/predict", json={"features": sample_features})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"PASS", "FAIL"}
    assert 0.0 <= body["probability"] <= 1.0
    contributions = body["top_contributing_features"]
    assert isinstance(contributions, list)
    assert len(contributions) == 5
    for entry in contributions:
        assert "feature" in entry
        assert "shap_value" in entry
        assert isinstance(entry["shap_value"], float)


def test_predict_rejects_wrong_feature_count(client):
    r = client.post("/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 422


def test_predict_rejects_empty_features(client):
    r = client.post("/predict", json={"features": []})
    assert r.status_code == 422


def test_drift_endpoint_returns_summary(client, sample_features):
    """Drift report on the first 50 wafers vs the training baseline."""
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "dashboard_assets", "data", "X_clean.csv"
    )
    batch = pd.read_csv(csv_path).head(50).values.tolist()
    r = client.post("/drift", json={"samples": batch})
    assert r.status_code == 200
    body = r.json()
    assert body["n_reference"] > 0
    assert body["n_current"] == 50
    assert body["n_significant"] >= 0
    assert isinstance(body["top_drifted_features"], list)
    if body["top_drifted_features"]:
        item = body["top_drifted_features"][0]
        assert {"feature", "psi", "severity"} <= set(item.keys())


def test_drift_endpoint_rejects_wrong_shape(client):
    r = client.post("/drift", json={"samples": [[1.0, 2.0]]})
    assert r.status_code == 422


def test_drift_endpoint_rejects_empty_batch(client):
    r = client.post("/drift", json={"samples": []})
    assert r.status_code == 422


def test_metadata_endpoint(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"]
    assert body["n_input_features"] > 0
    assert len(body["selected_feature_names"]) > 0
    assert isinstance(body["drift_reference_loaded"], bool)


def test_request_id_header_is_set(client):
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")
    assert len(r.headers["X-Request-ID"]) >= 8


def test_predict_batch_returns_per_sample_predictions(client, sample_features):
    r = client.post("/predict/batch", json={"samples": [sample_features, sample_features]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_samples"] == 2
    assert len(body["predictions"]) == 2
    for item in body["predictions"]:
        assert item["prediction"] in {"PASS", "FAIL"}
        assert 0.0 <= item["probability"] <= 1.0


def test_predict_batch_rejects_wrong_shape(client, sample_features):
    r = client.post("/predict/batch", json={"samples": [sample_features, [1.0, 2.0]]})
    assert r.status_code == 422


def test_predict_batch_rejects_empty(client):
    r = client.post("/predict/batch", json={"samples": []})
    assert r.status_code == 422
