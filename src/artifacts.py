"""
artifacts.py — Shared loader for trained pipeline artifacts.

Both the FastAPI service (`api/main.py`) and the dashboard's
DiagnosticsPipeline (`src/diagnostics.py`) need to load the same five
artifacts in the same order:

  preprocessing/var_selector.joblib
  preprocessing/scaler.joblib
  feature_engineering/corr_kept_cols.json
  feature_engineering/mi_selected_cols.json
  feature_names_input.json
  rf_model.joblib

Centralizing this avoids drift between the two call sites.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import joblib

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """All artifacts needed to run inference + SHAP on a single wafer."""
    var_selector: Any
    scaler: Any
    corr_kept_cols: list[str]
    mi_selected_cols: list[str]
    input_feature_names: list[str]
    model: Any

    @property
    def preprocess_artifacts(self) -> dict:
        """Dict format expected by `run_preprocessing_pipeline(..., fit=False)`."""
        return {"var_selector": self.var_selector, "scaler": self.scaler}


def load_pipeline_artifacts(model_dir: str) -> PipelineArtifacts:
    """
    Load every artifact produced by `src/train.py` from `model_dir`.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing rf_model.joblib, feature_names_input.json,
        preprocessing/, and feature_engineering/.

    Returns
    -------
    PipelineArtifacts dataclass holding all loaded objects.
    """
    pp_dir = os.path.join(model_dir, "preprocessing")
    fe_dir = os.path.join(model_dir, "feature_engineering")

    var_selector = joblib.load(os.path.join(pp_dir, "var_selector.joblib"))
    scaler = joblib.load(os.path.join(pp_dir, "scaler.joblib"))

    with open(os.path.join(fe_dir, "corr_kept_cols.json")) as f:
        corr_kept_cols = json.load(f)
    with open(os.path.join(fe_dir, "mi_selected_cols.json")) as f:
        mi_selected_cols = json.load(f)
    with open(os.path.join(model_dir, "feature_names_input.json")) as f:
        input_feature_names = json.load(f)

    model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))

    logger.info(
        "Loaded pipeline from %s: %d input feats -> %d after corr -> %d MI -> RF (%s trees)",
        model_dir,
        len(input_feature_names),
        len(corr_kept_cols),
        len(mi_selected_cols),
        getattr(model, "n_estimators", "?"),
    )

    return PipelineArtifacts(
        var_selector=var_selector,
        scaler=scaler,
        corr_kept_cols=corr_kept_cols,
        mi_selected_cols=mi_selected_cols,
        input_feature_names=input_feature_names,
        model=model,
    )
