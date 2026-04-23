"""
FastAPI endpoint for SECOM fault-detection model.

Takes 446 sensor readings for a single wafer and runs them through the
same feature pipeline used during training:
  446 sensors → remove low-variance → scale → remove correlated duplicates
  → select top 50 → Random Forest prediction → SHAP explanation

Endpoints:
  POST /predict  — returns pass/fail, probability, and top contributing sensors
  GET  /health   — liveness check with pipeline metadata
"""

import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.artifacts import load_pipeline_artifacts
from src.preprocess import run_preprocessing_pipeline

# ─── Paths ───────────────────────────────────────────────
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models")
)

# ─── Global state (loaded once at startup) ───────────────
model = None
explainer = None
preprocess_artifacts: dict = {}
input_feature_names: list[str] = []
corr_kept_cols: list[str] = []
mi_selected_cols: list[str] = []
N_INPUT_FEATURES: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all pipeline artifacts on startup."""
    global model, explainer, preprocess_artifacts
    global input_feature_names, corr_kept_cols, mi_selected_cols, N_INPUT_FEATURES

    artifacts = load_pipeline_artifacts(MODEL_DIR)
    preprocess_artifacts = artifacts.preprocess_artifacts
    corr_kept_cols = artifacts.corr_kept_cols
    mi_selected_cols = artifacts.mi_selected_cols
    input_feature_names = artifacts.input_feature_names
    N_INPUT_FEATURES = len(input_feature_names)
    model = artifacts.model
    explainer = shap.TreeExplainer(model)

    yield


app = FastAPI(
    title="SECOM Fault Detection API",
    description=(
        "Predicts whether a semiconductor wafer will pass or fail quality "
        "inspection, based on 446 sensor readings from the manufacturing "
        "process. Every prediction comes with a SHAP explanation showing "
        "which sensors contributed most."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response Schemas ──────────────────────────


class PredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_not_empty(cls, v):
        if not v:
            raise ValueError("features list must not be empty")
        return v


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    top_contributing_features: list[FeatureContribution]


# ─── Inference helpers ───────────────────────────────────


def transform_input(raw_features: list[float]) -> pd.DataFrame:
    """
    Apply the full NB02 feature pipeline to a single sample.

    446 raw values → variance filter → scale → corr filter → MI select → 50 features

    Returns a DataFrame (not ndarray) so sklearn sees the feature names
    it was trained with — avoids UserWarning on predict.
    """
    # Build DataFrame matching training input shape
    X_raw = pd.DataFrame([raw_features], columns=input_feature_names)

    # Preprocessing: variance filter + StandardScaler (inference mode)
    X_processed, _ = run_preprocessing_pipeline(
        X_raw, fit=False, artifacts=preprocess_artifacts
    )

    # Correlation filter: keep only columns that survived training
    X_decorr = X_processed[corr_kept_cols]

    # MI selection: keep only the top-50 features
    X_selected = X_decorr[mi_selected_cols]

    return X_selected


# ─── Endpoints ───────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipeline": {
            "input_features": N_INPUT_FEATURES,
            "after_corr_filter": len(corr_kept_cols),
            "after_mi_select": len(mi_selected_cols),
            "model_trees": model.n_estimators if model else 0,
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run the full pipeline + SHAP explanation on a single sample."""

    if len(req.features) != N_INPUT_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {N_INPUT_FEATURES} features, got {len(req.features)}",
        )

    # ── Transform through the full pipeline ──
    x = transform_input(req.features)

    # ── Predict ──
    prob_fail = float(model.predict_proba(x)[0, 1])
    label = "FAIL" if prob_fail >= 0.5 else "PASS"

    # ── SHAP (class-1 = Fail) ──
    shap_values = explainer.shap_values(x)

    if isinstance(shap_values, list):
        shap_fail = shap_values[1][0]       # old SHAP API
    else:
        shap_fail = shap_values[0, :, 1]    # new 3-D API

    # Top 5 features by absolute SHAP contribution
    top_k = 5
    top_idx = np.argsort(np.abs(shap_fail))[::-1][:top_k]
    top_features = [
        FeatureContribution(
            feature=mi_selected_cols[i],
            shap_value=round(float(shap_fail[i]), 6),
        )
        for i in top_idx
    ]

    return PredictResponse(
        prediction=label,
        probability=round(prob_fail, 4),
        top_contributing_features=top_features,
    )
