"""
diagnostics.py — Yield debug analysis engine for SECOM fault detection.

Provides baseline computation, per-sample deviation analysis, and
rule-based root-cause summary generation. Used by both the Streamlit
dashboard and the yield debug notebook (`05_yield_debug_analysis.ipynb`).

Design:
  - BaselineProfile stores pass-only population statistics
  - SampleDiagnostics holds per-sample deviation + SHAP results
  - generate_root_cause_summary produces grounded text (no LLM fluff)
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.preprocess import load_clean, run_preprocessing_pipeline


# ─── Data classes ─────────────────────────────────────────

@dataclass
class BaselineProfile:
    """Statistics from pass-only (healthy) samples for the MI-selected features."""
    mean: pd.Series
    std: pd.Series
    median: pd.Series
    q25: pd.Series
    q75: pd.Series
    n_samples: int


@dataclass
class SampleDiagnostics:
    """Full diagnostic result for a single wafer sample."""
    sample_index: int
    prediction: str
    probability: float
    feature_values: pd.Series
    shap_values: pd.Series
    deviations_z: pd.Series
    top_shap_features: List[str]
    top_deviated_features: List[str]
    combined_risk_features: List[str]


# ─── Pipeline loader ─────────────────────────────────────

class DiagnosticsPipeline:
    """
    Loads all trained artifacts and provides diagnostic methods.

    This mirrors the same inference pipeline as api/main.py but is designed
    for batch/interactive use in Streamlit and notebooks rather than HTTP.
    """

    def __init__(self, model_dir: str = "models", data_dir: str = "data/processed"):
        self.model_dir = model_dir
        self.data_dir = data_dir

        self.model = None
        self.explainer = None
        self.preprocess_artifacts = {}
        self.input_feature_names = []
        self.corr_kept_cols = []
        self.mi_selected_cols = []
        self.baseline: Optional[BaselineProfile] = None
        self._X_clean = None
        self._y = None

    def load(self):
        """Load model artifacts and compute baseline from training data."""
        pp_dir = os.path.join(self.model_dir, "preprocessing")
        self.preprocess_artifacts = {
            "var_selector": joblib.load(os.path.join(pp_dir, "var_selector.joblib")),
            "scaler": joblib.load(os.path.join(pp_dir, "scaler.joblib")),
        }

        fe_dir = os.path.join(self.model_dir, "feature_engineering")
        with open(os.path.join(fe_dir, "corr_kept_cols.json")) as f:
            self.corr_kept_cols = json.load(f)
        with open(os.path.join(fe_dir, "mi_selected_cols.json")) as f:
            self.mi_selected_cols = json.load(f)

        with open(os.path.join(self.model_dir, "feature_names_input.json")) as f:
            self.input_feature_names = json.load(f)

        self.model = joblib.load(os.path.join(self.model_dir, "rf_model.joblib"))
        self.explainer = shap.TreeExplainer(self.model)

        self._X_clean, self._y = load_clean(self.data_dir)
        self.baseline = self._compute_baseline()

        return self

    def _transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature pipeline: variance filter → scale → corr → MI select."""
        X_processed, _ = run_preprocessing_pipeline(
            X_raw, fit=False, artifacts=self.preprocess_artifacts
        )
        X_decorr = X_processed[self.corr_kept_cols]
        X_selected = X_decorr[self.mi_selected_cols]
        return X_selected

    def _compute_baseline(self) -> BaselineProfile:
        """Compute statistics from pass-only samples (healthy baseline)."""
        pass_mask = self._y == 0
        X_pass = self._X_clean[pass_mask]
        X_pass_transformed = self._transform(X_pass)

        return BaselineProfile(
            mean=X_pass_transformed.mean(),
            std=X_pass_transformed.std(),
            median=X_pass_transformed.median(),
            q25=X_pass_transformed.quantile(0.25),
            q75=X_pass_transformed.quantile(0.75),
            n_samples=int(pass_mask.sum()),
        )

    # ─── Public API ───────────────────────────────────────

    def get_sample(self, index: int) -> pd.Series:
        """Retrieve a single sample (446 raw cleaned features) by dataset index."""
        return self._X_clean.iloc[index]

    def get_label(self, index: int) -> int:
        """Get the actual label for a sample (0=pass, 1=fail)."""
        return int(self._y.iloc[index])

    @property
    def n_samples(self) -> int:
        return len(self._X_clean)

    @property
    def fail_indices(self) -> List[int]:
        return self._y[self._y == 1].index.tolist()

    @property
    def pass_indices(self) -> List[int]:
        return self._y[self._y == 0].index.tolist()

    def diagnose_sample(self, raw_features: pd.Series, sample_index: int = -1,
                        top_k: int = 10) -> SampleDiagnostics:
        """
        Run full diagnostics on a single sample.

        Parameters
        ----------
        raw_features : pd.Series — 446 cleaned sensor values
        sample_index : int — dataset row index (-1 if manual input)
        top_k : int — number of top features to report

        Returns
        -------
        SampleDiagnostics with prediction, SHAP, deviations, and ranked features
        """
        X_raw = pd.DataFrame([raw_features.values], columns=self.input_feature_names)
        X_selected = self._transform(X_raw)

        prob_fail = float(self.model.predict_proba(X_selected)[0, 1])
        label = "FAIL" if prob_fail >= 0.5 else "PASS"

        shap_raw = self.explainer.shap_values(X_selected)
        if isinstance(shap_raw, list):
            shap_fail = shap_raw[1][0]
        else:
            shap_fail = shap_raw[0, :, 1]

        shap_series = pd.Series(shap_fail, index=self.mi_selected_cols)
        feature_vals = X_selected.iloc[0]

        baseline_std = self.baseline.std.replace(0, 1e-8)
        z_scores = (feature_vals - self.baseline.mean) / baseline_std

        top_shap_idx = np.argsort(np.abs(shap_fail))[::-1][:top_k]
        top_shap_features = [self.mi_selected_cols[i] for i in top_shap_idx]

        top_dev_features = z_scores.abs().sort_values(ascending=False).head(top_k).index.tolist()

        shap_set = set(top_shap_features[:top_k])
        dev_set = set(top_dev_features[:top_k])
        combined = [f for f in top_shap_features if f in dev_set]
        combined += [f for f in top_dev_features if f not in shap_set and len(combined) < top_k]

        return SampleDiagnostics(
            sample_index=sample_index,
            prediction=label,
            probability=prob_fail,
            feature_values=feature_vals,
            shap_values=shap_series,
            deviations_z=z_scores,
            top_shap_features=top_shap_features,
            top_deviated_features=top_dev_features,
            combined_risk_features=combined,
        )

    def diagnose_by_index(self, index: int, top_k: int = 10) -> SampleDiagnostics:
        """Convenience: diagnose a sample by its dataset index."""
        sample = self.get_sample(index)
        return self.diagnose_sample(sample, sample_index=index, top_k=top_k)

    def batch_diagnose(self, indices: List[int], top_k: int = 10) -> List[SampleDiagnostics]:
        """Diagnose multiple samples (for notebook analysis)."""
        return [self.diagnose_by_index(i, top_k=top_k) for i in indices]

    def get_baseline_comparison_df(self, diag: SampleDiagnostics,
                                    features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Build a comparison table: sample value vs baseline stats for selected features.
        """
        if features is None:
            features = diag.combined_risk_features[:10]

        rows = []
        for feat in features:
            rows.append({
                "Sensor": feat,
                "Sample Value": round(float(diag.feature_values[feat]), 4),
                "Baseline Mean": round(float(self.baseline.mean[feat]), 4),
                "Baseline Std": round(float(self.baseline.std[feat]), 4),
                "Z-Score": round(float(diag.deviations_z[feat]), 2),
                "|SHAP|": round(float(abs(diag.shap_values[feat])), 6),
            })
        return pd.DataFrame(rows)

    def get_failure_pattern_summary(self, indices: Optional[List[int]] = None,
                                     top_k: int = 10) -> pd.DataFrame:
        """
        Aggregate top sensor drivers across multiple failed samples.
        Shows how often each sensor appears as a top contributor.
        """
        if indices is None:
            indices = self.fail_indices

        feature_counts = pd.Series(dtype=int)
        shap_accumulator = pd.Series(0.0, index=self.mi_selected_cols)
        n = 0

        for idx in indices:
            try:
                diag = self.diagnose_by_index(idx, top_k=top_k)
                for feat in diag.top_shap_features[:top_k]:
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
                shap_accumulator += diag.shap_values.abs()
                n += 1
            except Exception:
                continue

        if n == 0:
            return pd.DataFrame()

        summary = pd.DataFrame({
            "Times in Top-{0}".format(top_k): feature_counts,
            "Mean |SHAP|": (shap_accumulator / n),
        }).sort_values("Mean |SHAP|", ascending=False).head(top_k)

        summary["Frequency %"] = (summary["Times in Top-{0}".format(top_k)] / n * 100).round(1)
        return summary


# ─── Root-cause summary generation ───────────────────────

def generate_root_cause_summary(diag: SampleDiagnostics, baseline: BaselineProfile,
                                 top_k: int = 5) -> str:
    """
    Generate a grounded, rule-based root-cause summary.

    Not LLM-generated — this is deterministic text built from actual
    SHAP values, z-scores, and feature rankings. The summary is designed
    to read like an engineer's preliminary failure analysis note.
    """
    prob_pct = diag.probability * 100

    if diag.prediction == "PASS":
        confidence = "low risk" if diag.probability < 0.2 else "marginal"
        header = (
            f"**Sample #{diag.sample_index}** is predicted as **PASS** "
            f"({confidence}, {prob_pct:.1f}% failure probability)."
        )
    else:
        confidence = "high risk" if diag.probability > 0.8 else "elevated risk"
        header = (
            f"**Sample #{diag.sample_index}** is predicted as **FAIL** "
            f"({confidence}, {prob_pct:.1f}% failure probability)."
        )

    top_shap = diag.top_shap_features[:top_k]
    top_deviated = diag.top_deviated_features[:top_k]
    overlap = [f for f in top_shap if f in set(top_deviated)]

    shap_list = ", ".join(top_shap[:3])
    driver_line = f"Top model drivers: **{shap_list}**."

    if overlap:
        overlap_names = ", ".join(overlap[:3])
        deviation_line = (
            f"Sensors **{overlap_names}** are both top model contributors and "
            f"show significant deviation from the healthy production baseline."
        )
    else:
        dev_list = ", ".join(top_deviated[:3])
        deviation_line = (
            f"Most deviated sensors ({dev_list}) differ from the top model "
            f"drivers, suggesting the deviations may not directly cause failure "
            f"but warrant process review."
        )

    detail_lines = []
    for feat in (overlap if overlap else top_shap[:3]):
        z = diag.deviations_z.get(feat, 0)
        val = diag.feature_values.get(feat, 0)
        mean = baseline.mean.get(feat, 0)
        direction = "above" if z > 0 else "below"
        detail_lines.append(
            f"- **{feat}**: value {val:.3f} ({abs(z):.1f}σ {direction} baseline mean {mean:.3f})"
        )
    detail_block = "\n".join(detail_lines)

    if diag.prediction == "FAIL" and overlap:
        action = (
            "**Recommended action:** Investigate the process steps associated with "
            f"{', '.join(overlap[:3])} for potential excursion or tool drift."
        )
    elif diag.prediction == "FAIL":
        action = (
            "**Recommended action:** Review recent process logs for anomalies. "
            "The model's top drivers may reflect indirect effects of an upstream issue."
        )
    else:
        action = "No immediate action required. Monitor these sensors on subsequent lots."

    return f"{header}\n\n{driver_line}\n\n{deviation_line}\n\n{detail_block}\n\n{action}"
