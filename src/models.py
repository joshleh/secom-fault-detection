"""
models.py — Model training, calibration, and threshold-tuning helpers.

Split out from src/train.py so the same building blocks can be reused
from notebooks (notably 03_model_comparison.ipynb) and exercised in tests.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

DEFAULT_RF_KWARGS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
)


@dataclass
class CVMetrics:
    """Mean ± std of held-out metrics across stratified K folds."""
    n_splits: int
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    fail_f1_mean: float
    fail_f1_std: float

    def as_dict(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "roc_auc_mean": round(self.roc_auc_mean, 4),
            "roc_auc_std": round(self.roc_auc_std, 4),
            "pr_auc_mean": round(self.pr_auc_mean, 4),
            "pr_auc_std": round(self.pr_auc_std, 4),
            "fail_f1_mean": round(self.fail_f1_mean, 4),
            "fail_f1_std": round(self.fail_f1_std, 4),
        }


def build_random_forest(random_state: int = 42, **overrides) -> RandomForestClassifier:
    """RF with the project's default class-balanced hyperparameters."""
    kwargs = {**DEFAULT_RF_KWARGS, "random_state": random_state, **overrides}
    return RandomForestClassifier(**kwargs)


def calibrate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """
    Probability calibration. RF probabilities are notoriously poorly calibrated
    (they cluster near 0 / 1) and the rest of the pipeline relies on a meaningful
    `predict_proba` (threshold tuning, dashboard's failure-probability metric).

    Wraps an *unfitted* estimator and fits via internal CV; for already-fitted
    estimators pass `cv="prefit"` and a held-out (X, y) calibration set.
    """
    cal = CalibratedClassifierCV(estimator, method=method, cv=cv)
    cal.fit(X, y)
    logger.info("Calibrated %s with method=%s cv=%s", type(estimator).__name__, method, cv)
    return cal


def cross_validated_metrics(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVMetrics:
    """
    Stratified K-fold CV metrics. With only ~104 fail samples, a single 80/20
    split is high variance; mean ± std across folds gives a more honest
    sense of model quality.

    `estimator_factory` should be a zero-arg callable that returns a fresh
    unfitted estimator each call (so each fold trains independently).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    roc_aucs, pr_aucs, f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        est = estimator_factory()
        est.fit(X_tr, y_tr)
        prob = est.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        roc_aucs.append(roc_auc_score(y_te, prob))
        pr_aucs.append(average_precision_score(y_te, prob))
        f1s.append(f1_score(y_te, pred, pos_label=1, zero_division=0))

        logger.info(
            "Fold %d/%d: ROC-AUC=%.3f  PR-AUC=%.3f  Fail-F1=%.3f",
            fold, n_splits, roc_aucs[-1], pr_aucs[-1], f1s[-1],
        )

    return CVMetrics(
        n_splits=n_splits,
        roc_auc_mean=float(np.mean(roc_aucs)),
        roc_auc_std=float(np.std(roc_aucs)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs)),
        fail_f1_mean=float(np.mean(f1s)),
        fail_f1_std=float(np.std(f1s)),
    )


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    objective: str = "f1",
    min_recall: float | None = None,
) -> tuple[float, dict]:
    """
    Choose a decision threshold from a precision-recall sweep.

    Parameters
    ----------
    y_true     : ground-truth binary labels
    y_prob     : positive-class probabilities
    objective  : "f1" (max F1) or "youden" (max precision*recall product)
    min_recall : if set, restricts the search to thresholds with recall >= floor
                 (use this in production to never miss more than X% of failures)

    Returns
    -------
    threshold  : float
    diagnostics: {precision, recall, f1, n_candidates}
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one extra precision/recall vs threshold
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0.0,
    )

    if min_recall is not None:
        mask = recalls >= min_recall
        if not mask.any():
            logger.warning(
                "No threshold reaches recall >= %.2f; falling back to unconstrained search.",
                min_recall,
            )
        else:
            precisions, recalls, thresholds, f1s = (
                precisions[mask], recalls[mask], thresholds[mask], f1s[mask]
            )

    if objective == "f1":
        best_idx = int(np.argmax(f1s))
    elif objective == "youden":
        best_idx = int(np.argmax(precisions * recalls))
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    threshold = float(thresholds[best_idx])
    diagnostics = {
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
        "objective": objective,
        "min_recall": min_recall,
        "n_candidates": int(len(thresholds)),
    }
    logger.info(
        "Best threshold (%s): %.3f -> precision=%.3f recall=%.3f f1=%.3f",
        objective, threshold, diagnostics["precision"],
        diagnostics["recall"], diagnostics["f1"],
    )
    return threshold, diagnostics


def save_threshold(threshold: float, diagnostics: dict, model_dir: str) -> str:
    """Persist tuned threshold + the metrics that justify it."""
    path = os.path.join(model_dir, "threshold.json")
    with open(path, "w") as f:
        json.dump({"threshold": float(threshold), **diagnostics}, f, indent=2)
    logger.info("Saved threshold to %s", path)
    return path


def load_threshold(model_dir: str, default: float = 0.5) -> float:
    """Read the tuned threshold; fall back to 0.5 if the file is missing."""
    path = os.path.join(model_dir, "threshold.json")
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return float(json.load(f).get("threshold", default))
