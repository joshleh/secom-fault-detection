"""
Smoke test for the SECOM API (notebook 02 pipeline).

Usage:
    1. Start the API:  uvicorn api.main:app --reload
    2. Run this:       python tests/test_api.py
"""

import requests
import pandas as pd

API_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{API_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    pipe = data["pipeline"]
    print(f"✓ /health → {pipe['input_features']} input → "
          f"{pipe['after_corr_filter']} corr → "
          f"{pipe['after_mi_select']} MI → "
          f"RF ({pipe['model_trees']} trees)")


def test_predict():
    # Grab first sample from the dataset (446 features)
    X = pd.read_csv("data/processed/X_clean.csv")
    sample = X.iloc[0].tolist()

    r = requests.post(f"{API_URL}/predict", json={"features": sample})
    assert r.status_code == 200

    data = r.json()
    print(f"✓ /predict →")
    print(f"  prediction: {data['prediction']}")
    print(f"  probability: {data['probability']}")
    print(f"  top features:")
    for feat in data["top_contributing_features"]:
        print(f"    {feat['feature']:>14s}  SHAP={feat['shap_value']:+.6f}")


def test_wrong_feature_count():
    """API should reject input with wrong number of features."""
    r = requests.post(f"{API_URL}/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 422
    print(f"✓ Wrong feature count → 422 (expected)")


def test_empty_features():
    """API should reject empty feature list."""
    r = requests.post(f"{API_URL}/predict", json={"features": []})
    assert r.status_code == 422
    print(f"✓ Empty features → 422 (expected)")


if __name__ == "__main__":
    test_health()
    test_predict()
    test_wrong_feature_count()
    test_empty_features()
    print("\nAll tests passed.")
