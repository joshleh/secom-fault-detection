"""
FastAPI endpoint for SECOM fault-detection model.

Takes 446 sensor readings for a single wafer (or a batch of them) and runs
them through the same feature pipeline used during training:
  446 sensors → remove low-variance → scale → remove correlated duplicates
  → select top 50 → Random Forest prediction → SHAP explanation

Endpoints:
  GET  /health    — liveness check with pipeline metadata
  GET  /metadata  — full pipeline metadata (feature names, model + threshold info)
  POST /predict   — single-wafer prediction with SHAP explanation
  POST /predict/batch — predict many wafers in one call (no SHAP, optimized)
  POST /drift     — PSI drift report vs the training-time baseline
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, field_validator

from src.artifacts import load_pipeline_artifacts
from src.drift import compute_drift_from_summary, load_reference_summary
from src.preprocess import run_preprocessing_pipeline

# ─── Logging (structured, single-line key=value records) ─
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("secom.api")

# ─── Paths ───────────────────────────────────────────────
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models")
)

# CORS_ORIGINS: comma-separated origins, or "*" to allow all (default for dev)
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))

# ─── Global state (loaded once at startup) ───────────────
model = None
explainer = None
preprocess_artifacts: dict = {}
input_feature_names: list[str] = []
corr_kept_cols: list[str] = []
mi_selected_cols: list[str] = []
N_INPUT_FEATURES: int = 0
THRESHOLD: float = 0.5
drift_reference_summary: dict | None = None
MODEL_VERSION: str = os.environ.get("MODEL_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all pipeline artifacts on startup."""
    global model, explainer, preprocess_artifacts
    global input_feature_names, corr_kept_cols, mi_selected_cols
    global N_INPUT_FEATURES, THRESHOLD, drift_reference_summary

    artifacts = load_pipeline_artifacts(MODEL_DIR)
    preprocess_artifacts = artifacts.preprocess_artifacts
    corr_kept_cols = artifacts.corr_kept_cols
    mi_selected_cols = artifacts.mi_selected_cols
    input_feature_names = artifacts.input_feature_names
    N_INPUT_FEATURES = len(input_feature_names)
    model = artifacts.model
    THRESHOLD = artifacts.threshold
    explainer = shap.TreeExplainer(model)

    drift_reference_summary = load_reference_summary(
        os.path.join(MODEL_DIR, "drift_reference.json")
    )

    logger.info(
        'event=startup model_dir="%s" n_input_features=%d threshold=%.4f drift_ref=%s',
        MODEL_DIR, N_INPUT_FEATURES, THRESHOLD,
        "loaded" if drift_reference_summary else "missing",
    )

    yield

    logger.info("event=shutdown")


app = FastAPI(
    title="SECOM Fault Detection API",
    description=(
        "Predicts whether a semiconductor wafer will pass or fail quality "
        "inspection, based on 446 sensor readings from the manufacturing "
        "process. Every prediction comes with a SHAP explanation showing "
        "which sensors contributed most."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Attach a request id, log method/path/status/latency in key=value form."""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        'event=request request_id=%s method=%s path=%s status=%d latency_ms=%.1f',
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


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


class BatchPredictRequest(BaseModel):
    """Up to MAX_BATCH_SIZE wafers per call."""
    samples: list[list[float]]

    @field_validator("samples")
    @classmethod
    def check_size(cls, v):
        if not v:
            raise ValueError("samples list must not be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"batch too large: {len(v)} > MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
            )
        return v


class BatchPredictItem(BaseModel):
    prediction: str
    probability: float


class BatchPredictResponse(BaseModel):
    n_samples: int
    n_predicted_fail: int
    threshold: float
    predictions: list[BatchPredictItem]


class DriftRequest(BaseModel):
    """Recent batch of wafers as a list of feature vectors (446 each)."""
    samples: list[list[float]]

    @field_validator("samples")
    @classmethod
    def check_not_empty(cls, v):
        if not v:
            raise ValueError("samples list must not be empty")
        return v


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    severity: str


class DriftResponse(BaseModel):
    n_reference: int
    n_current: int
    n_significant: int
    n_moderate: int
    top_drifted_features: list[FeatureDrift]


class MetadataResponse(BaseModel):
    # Disable Pydantic's "model_" protected-namespace warning so we can keep
    # the natural field names model_version / model_type / model_n_estimators.
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    n_input_features: int
    n_after_corr_filter: int
    n_after_mi_select: int
    decision_threshold: float
    model_type: str
    model_n_estimators: int | None
    drift_reference_loaded: bool
    drift_reference_n_samples: int | None
    input_feature_names_sample: list[str]
    selected_feature_names: list[str]


# ─── Inference helpers ───────────────────────────────────


def _validate_feature_count(received: int, where: str) -> None:
    if received != N_INPUT_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"{where}: expected {N_INPUT_FEATURES} features, got {received}",
        )


def transform_input(raw_features: list[float]) -> pd.DataFrame:
    """Apply the full pipeline to a single sample (returns 1x50 DataFrame)."""
    return transform_batch([raw_features])


def transform_batch(rows: list[list[float]]) -> pd.DataFrame:
    """Vectorized pipeline: list of 446-vectors → N x 50 DataFrame."""
    X_raw = pd.DataFrame(rows, columns=input_feature_names)
    X_processed, _ = run_preprocessing_pipeline(
        X_raw, fit=False, artifacts=preprocess_artifacts
    )
    return X_processed[corr_kept_cols][mi_selected_cols]


# ─── Endpoints ───────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipeline": {
            "input_features": N_INPUT_FEATURES,
            "after_corr_filter": len(corr_kept_cols),
            "after_mi_select": len(mi_selected_cols),
            "model_trees": getattr(model, "n_estimators", 0) if model else 0,
            "decision_threshold": round(THRESHOLD, 4),
        },
    }


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    """Full pipeline metadata. Useful for clients constructing valid payloads."""
    return MetadataResponse(
        model_version=MODEL_VERSION,
        n_input_features=N_INPUT_FEATURES,
        n_after_corr_filter=len(corr_kept_cols),
        n_after_mi_select=len(mi_selected_cols),
        decision_threshold=round(THRESHOLD, 4),
        model_type=type(model).__name__ if model else "unknown",
        model_n_estimators=getattr(model, "n_estimators", None),
        drift_reference_loaded=drift_reference_summary is not None,
        drift_reference_n_samples=(
            int(drift_reference_summary.get("n_samples", 0))
            if drift_reference_summary else None
        ),
        input_feature_names_sample=input_feature_names[:5],
        selected_feature_names=mi_selected_cols,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run the full pipeline + SHAP explanation on a single sample."""
    _validate_feature_count(len(req.features), where="features")

    x = transform_input(req.features)
    prob_fail = float(model.predict_proba(x)[0, 1])
    label = "FAIL" if prob_fail >= THRESHOLD else "PASS"

    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        shap_fail = shap_values[1][0]       # old SHAP API
    else:
        shap_fail = shap_values[0, :, 1]    # new 3-D API

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


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    """
    Predict on a batch of wafers in a single call. SHAP is intentionally
    skipped here so the endpoint stays fast for high-throughput scenarios
    (e.g. scoring an entire production day). Use /predict for SHAP.
    """
    bad = [i for i, row in enumerate(req.samples) if len(row) != N_INPUT_FEATURES]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Each sample must have {N_INPUT_FEATURES} features; "
                f"sample(s) {bad[:5]}{'...' if len(bad) > 5 else ''} did not."
            ),
        )

    X = transform_batch(req.samples)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    return BatchPredictResponse(
        n_samples=len(req.samples),
        n_predicted_fail=int(preds.sum()),
        threshold=round(THRESHOLD, 4),
        predictions=[
            BatchPredictItem(
                prediction="FAIL" if p == 1 else "PASS",
                probability=round(float(prob), 4),
            )
            for p, prob in zip(preds, probs, strict=True)
        ],
    )


@app.post("/drift", response_model=DriftResponse)
def drift(req: DriftRequest):
    """
    PSI drift report for a recent batch vs the training-time baseline.

    Each row in `samples` must be a 446-feature vector (same input shape
    as /predict). Returns the top-10 features by PSI; the full per-feature
    table is computed but only the largest movers are returned to keep
    the response small.
    """
    if drift_reference_summary is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Drift reference not available. Retrain with the latest "
                "src/train.py to produce models/drift_reference.json."
            ),
        )

    bad = [i for i, row in enumerate(req.samples) if len(row) != N_INPUT_FEATURES]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Each sample must have {N_INPUT_FEATURES} features; "
                f"sample(s) {bad[:5]}{'...' if len(bad) > 5 else ''} did not."
            ),
        )

    current = pd.DataFrame(req.samples, columns=input_feature_names)
    report = compute_drift_from_summary(drift_reference_summary, current)

    top = report.per_feature.head(10).reset_index()
    return DriftResponse(
        n_reference=report.n_reference,
        n_current=report.n_current,
        n_significant=report.n_significant,
        n_moderate=report.n_moderate,
        top_drifted_features=[
            FeatureDrift(
                feature=row["feature"],
                psi=float(row["psi"]) if not pd.isna(row["psi"]) else 0.0,
                severity=row["severity"],
            )
            for _, row in top.iterrows()
        ],
    )
