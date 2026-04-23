"""
drift.py — Population stability and distribution-shift checks.

Two complementary tests per feature:

  * Population Stability Index (PSI) — a binned KL-style score widely
    used in credit risk + manufacturing for monitoring whether a
    feature's distribution has shifted vs a reference (training)
    population. Conventional thresholds:
      PSI < 0.1   no significant drift
      0.1-0.25    moderate drift, investigate
      > 0.25      significant drift, retrain

  * Two-sample Kolmogorov-Smirnov test — non-parametric, returns a
    p-value for the null "samples come from the same distribution".

We compare *raw cleaned* features (post-EDA, pre-pipeline) so the
report is interpretable in original sensor units.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_BUCKETS_DEFAULT = 10
PSI_THRESHOLDS = {"none": 0.10, "moderate": 0.25}


def _safe_quantile_bins(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based bin edges from the reference distribution.

    Falls back to uniform edges if the reference has too many ties for
    quantile bucketing to produce distinct cuts.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        lo, hi = float(np.min(reference)), float(np.max(reference))
        if lo == hi:
            return np.array([lo - 0.5, lo + 0.5])
        edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = PSI_BUCKETS_DEFAULT,
    epsilon: float = 1e-6,
) -> float:
    """
    PSI = sum_i (current_i - reference_i) * ln(current_i / reference_i)

    where the sums run over `n_bins` quantile-defined bins of the
    reference distribution. Symmetric and >= 0.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = _safe_quantile_bins(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def classify_psi(psi: float) -> str:
    """Bucket a PSI value into a textual severity."""
    if np.isnan(psi):
        return "unknown"
    if psi < PSI_THRESHOLDS["none"]:
        return "none"
    if psi < PSI_THRESHOLDS["moderate"]:
        return "moderate"
    return "significant"


@dataclass
class DriftReport:
    """Per-feature drift summary."""
    n_reference: int
    n_current: int
    per_feature: pd.DataFrame  # columns: psi, severity, ks_stat, ks_pvalue
    n_significant: int
    n_moderate: int

    def as_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "n_significant": self.n_significant,
            "n_moderate": self.n_moderate,
            "top_drifted_features": self.per_feature.head(10).reset_index().to_dict("records"),
        }


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = PSI_BUCKETS_DEFAULT,
) -> DriftReport:
    """
    Compute PSI + KS test for every column shared by reference and current.

    Parameters
    ----------
    reference : pd.DataFrame — historical baseline (e.g. training data)
    current   : pd.DataFrame — recent batch to check for drift
    n_bins    : int — number of quantile bins for PSI

    Returns
    -------
    DriftReport with per-feature stats sorted by PSI descending.
    """
    shared_cols = [c for c in reference.columns if c in current.columns]
    if not shared_cols:
        raise ValueError("Reference and current frames share no columns.")

    rows = []
    for col in shared_cols:
        ref_vals = reference[col].to_numpy(dtype=float)
        cur_vals = current[col].to_numpy(dtype=float)

        psi = population_stability_index(ref_vals, cur_vals, n_bins=n_bins)

        ref_clean = ref_vals[~np.isnan(ref_vals)]
        cur_clean = cur_vals[~np.isnan(cur_vals)]
        if len(ref_clean) > 1 and len(cur_clean) > 1:
            ks_stat, ks_p = stats.ks_2samp(ref_clean, cur_clean)
        else:
            ks_stat, ks_p = float("nan"), float("nan")

        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "severity": classify_psi(psi),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
        })

    df = pd.DataFrame(rows).set_index("feature")
    df = df.sort_values("psi", ascending=False)

    return DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        per_feature=df,
        n_significant=int((df["severity"] == "significant").sum()),
        n_moderate=int((df["severity"] == "moderate").sum()),
    )


def save_reference_summary(reference: pd.DataFrame, path: str) -> None:
    """
    Persist a compact summary of the reference distribution that's enough
    to reconstruct PSI later (deciles per feature) without shipping the
    raw data alongside the model.
    """
    deciles = {
        col: reference[col].quantile(np.linspace(0, 1, 11)).round(6).tolist()
        for col in reference.columns
    }
    payload = {
        "n_samples": int(len(reference)),
        "feature_deciles": deciles,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    logger.info("Saved drift reference summary (%d features) to %s", len(deciles), path)


def load_reference_summary(path: str) -> dict | None:
    """Load the reference summary saved by `save_reference_summary`."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _psi_from_edges(
    edges: np.ndarray,
    reference_pct: np.ndarray,
    current: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """PSI when the reference's per-bin proportions are precomputed."""
    current = current[~np.isnan(current)]
    if len(current) == 0:
        return float("nan")
    cur_counts, _ = np.histogram(current, bins=edges)
    cur_pct = cur_counts / max(len(current), 1) + epsilon
    ref_pct = reference_pct + epsilon
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_from_summary(
    summary: dict,
    current: pd.DataFrame,
) -> DriftReport:
    """
    PSI-only drift report using a previously saved reference summary.

    The summary stores deciles per feature, which are enough to recover
    the bin edges used during training (decile binning => 10% per bin).
    KS p-values are not available without the raw reference data, so
    they are returned as NaN.
    """
    feature_deciles = summary.get("feature_deciles", {})
    n_reference = int(summary.get("n_samples", 0))
    if not feature_deciles:
        raise ValueError("Reference summary has no feature_deciles.")

    # Each feature has 10 equal-frequency bins => uniform 0.1 reference proportions
    ref_pct = np.full(10, 0.1)

    rows = []
    for col, deciles in feature_deciles.items():
        if col not in current.columns:
            continue
        edges = np.unique(np.asarray(deciles, dtype=float))
        if len(edges) < 3:
            psi = float("nan")
        else:
            edges = edges.copy()
            edges[0] = -np.inf
            edges[-1] = np.inf
            ref_pct_used = ref_pct[: len(edges) - 1]
            ref_pct_used = ref_pct_used / ref_pct_used.sum()
            psi = _psi_from_edges(
                edges, ref_pct_used, current[col].to_numpy(dtype=float)
            )
        rows.append({
            "feature": col,
            "psi": round(psi, 4) if not np.isnan(psi) else float("nan"),
            "severity": classify_psi(psi),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
        })

    df = pd.DataFrame(rows).set_index("feature").sort_values("psi", ascending=False)

    return DriftReport(
        n_reference=n_reference,
        n_current=len(current),
        per_feature=df,
        n_significant=int((df["severity"] == "significant").sum()),
        n_moderate=int((df["severity"] == "moderate").sum()),
    )
