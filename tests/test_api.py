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
