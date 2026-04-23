"""
Train the Random Forest model using the notebook 02 pipeline.

Pipeline (matches 02_modeling.ipynb exactly):
  1. load_clean()                            → 446 features (EDA-cleaned)
  2. run_preprocessing_pipeline()            → variance filter + StandardScaler
  3. drop_highly_correlated(threshold=0.95)  → remove redundant sensors
  4. select_top_k_by_mutual_info(k=50)       → MI-based feature selection
  5. Stratified 5-fold CV (mean ± std reported)
  6. RandomForestClassifier(balanced) trained on a stratified train split
  7. F1-optimal threshold tuned on the held-out validation fold

Note: `src/models.py` also exposes a `calibrate()` helper (isotonic /
sigmoid) for use from notebooks. It is intentionally NOT applied here
because (a) SHAP's TreeExplainer needs the bare RF and (b) with ~83 fail
samples in train, calibration is unreliable.

Saves to models/:
  preprocessing/var_selector.joblib        (fitted VarianceThreshold)
  preprocessing/scaler.joblib              (fitted StandardScaler)
  feature_engineering/corr_kept_cols.json   (columns surviving correlation filter)
  feature_engineering/mi_selected_cols.json (top-50 MI feature names)
  rf_model.joblib                          (fitted Random Forest)
  feature_names_input.json                 (original 446 input column names)
  threshold.json                           (F1-optimal decision threshold + metrics)
  cv_metrics.json                          (cross-validated ROC-AUC / PR-AUC / F1)
"""

import json
import logging
import os
import sys

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import drop_highly_correlated, select_top_k_by_mutual_info
from src.models import (
    build_random_forest,
    cross_validated_metrics,
    find_best_threshold,
    save_threshold,
)
from src.preprocess import load_clean, run_preprocessing_pipeline, save_artifacts

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_and_save(data_dir: str = "data/processed") -> None:
    """End-to-end: load → engineer → CV → train+calibrate → tune → evaluate → save."""

    # ── Step 1: Load EDA-cleaned data ──
    X_clean, y = load_clean(data_dir)
    input_feature_names = X_clean.columns.tolist()
    logger.info(
        "Loaded %d samples x %d features  |  Fail rate: %.1f%%",
        X_clean.shape[0], X_clean.shape[1], y.mean() * 100,
    )

    # ── Step 2: Variance filter + StandardScaler ──
    X_processed, preprocess_artifacts = run_preprocessing_pipeline(
        X_clean, variance_threshold=0.0
    )

    # ── Step 3: Correlation filter ──
    X_decorr, _ = drop_highly_correlated(X_processed, threshold=0.95)
    corr_kept_cols = X_decorr.columns.tolist()

    # ── Step 4: Mutual information selection (top 50) ──
    X_selected, _ = select_top_k_by_mutual_info(
        X_decorr, y, k=50, random_state=RANDOM_STATE
    )
    mi_selected_cols = X_selected.columns.tolist()

    # ── Step 5: Stratified 5-fold CV (mean ± std) ──
    logger.info("\n-- Stratified 5-fold cross-validation --")
    cv_metrics = cross_validated_metrics(
        lambda: build_random_forest(random_state=RANDOM_STATE),
        X_selected, y, n_splits=5, random_state=RANDOM_STATE,
    )
    logger.info(
        "CV summary: ROC-AUC=%.3f±%.3f  PR-AUC=%.3f±%.3f  Fail-F1=%.3f±%.3f",
        cv_metrics.roc_auc_mean, cv_metrics.roc_auc_std,
        cv_metrics.pr_auc_mean, cv_metrics.pr_auc_std,
        cv_metrics.fail_f1_mean, cv_metrics.fail_f1_std,
    )

    # ── Step 6: Train RF on a stratified train/val split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info(
        "\nTrain: %d samples (%d fails)  |  Val: %d samples (%d fails)",
        X_train.shape[0], int(y_train.sum()), X_val.shape[0], int(y_val.sum()),
    )

    rf = build_random_forest(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    # ── Step 7: Tune decision threshold on the validation set ──
    val_prob = rf.predict_proba(X_val)[:, 1]
    threshold, threshold_diag = find_best_threshold(y_val, val_prob, objective="f1")

    val_pred = (val_prob >= threshold).astype(int)
    logger.info("\n-- Validation Set Performance @ tuned threshold --")
    logger.info("\n%s", classification_report(y_val, val_pred, target_names=["Pass", "Fail"]))
    logger.info("ROC-AUC: %.4f", roc_auc_score(y_val, val_prob))

    # ── Save all artifacts ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_artifacts(preprocess_artifacts, save_dir=os.path.join(MODEL_DIR, "preprocessing"))

    fe_dir = os.path.join(MODEL_DIR, "feature_engineering")
    os.makedirs(fe_dir, exist_ok=True)
    with open(os.path.join(fe_dir, "corr_kept_cols.json"), "w") as f:
        json.dump(corr_kept_cols, f)
    with open(os.path.join(fe_dir, "mi_selected_cols.json"), "w") as f:
        json.dump(mi_selected_cols, f)

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_names_input.json"), "w") as f:
        json.dump(input_feature_names, f)

    save_threshold(threshold, threshold_diag, MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "cv_metrics.json"), "w") as f:
        json.dump(cv_metrics.as_dict(), f, indent=2)

    logger.info("\nAll artifacts saved to %s/", MODEL_DIR)


if __name__ == "__main__":
    train_and_save()
