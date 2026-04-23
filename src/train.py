"""
Train the Random Forest model using the notebook 02 pipeline.

Pipeline (matches 02_modeling.ipynb exactly):
  1. load_clean()                            → 446 features (EDA-cleaned)
  2. run_preprocessing_pipeline()            → variance filter + StandardScaler
  3. drop_highly_correlated(threshold=0.95)  → remove redundant sensors
  4. select_top_k_by_mutual_info(k=50)       → MI-based feature selection
  5. train/test split (stratified)
  6. RandomForestClassifier(balanced)

Saves to models/:
  preprocessing/var_selector.joblib        (fitted VarianceThreshold)
  preprocessing/scaler.joblib              (fitted StandardScaler)
  feature_engineering/corr_kept_cols.json   (columns surviving correlation filter)
  feature_engineering/mi_selected_cols.json (top-50 MI feature names)
  rf_model.joblib                          (fitted classifier)
  feature_names_input.json                 (original 446 input column names)
"""

import json
import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import drop_highly_correlated, select_top_k_by_mutual_info
from src.preprocess import load_clean, run_preprocessing_pipeline, save_artifacts

RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_and_save(data_dir: str = "data/processed") -> None:
    """End-to-end: load → engineer → train → evaluate → save."""

    # ── Step 1: Load EDA-cleaned data ──
    X_clean, y = load_clean(data_dir)
    input_feature_names = X_clean.columns.tolist()
    print(f"Loaded {X_clean.shape[0]} samples x {X_clean.shape[1]} features  |  "
          f"Fail rate: {y.mean():.1%}")

    # ── Step 2: Variance filter + StandardScaler ──
    X_processed, preprocess_artifacts = run_preprocessing_pipeline(
        X_clean, variance_threshold=0.0
    )

    # ── Step 3: Correlation filter ──
    X_decorr, dropped_corr = drop_highly_correlated(X_processed, threshold=0.95)
    corr_kept_cols = X_decorr.columns.tolist()

    # ── Step 4: Mutual information selection (top 50) ──
    X_selected, mi_scores = select_top_k_by_mutual_info(
        X_decorr, y, k=50, random_state=RANDOM_STATE
    )
    mi_selected_cols = X_selected.columns.tolist()

    # ── Step 5: Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\nTrain: {X_train.shape[0]} samples ({y_train.sum()} fails)")
    print(f"Test:  {X_test.shape[0]} samples ({y_test.sum()} fails)")

    # ── Step 6: Train RF ──
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n-- Test Set Performance --")
    print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # ── Save all artifacts ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Preprocessing transformers (var_selector + scaler)
    save_artifacts(preprocess_artifacts, save_dir=os.path.join(MODEL_DIR, "preprocessing"))

    # Feature engineering column lists
    fe_dir = os.path.join(MODEL_DIR, "feature_engineering")
    os.makedirs(fe_dir, exist_ok=True)

    with open(os.path.join(fe_dir, "corr_kept_cols.json"), "w") as f:
        json.dump(corr_kept_cols, f)

    with open(os.path.join(fe_dir, "mi_selected_cols.json"), "w") as f:
        json.dump(mi_selected_cols, f)

    # Model
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))

    # Input feature names (for API validation)
    with open(os.path.join(MODEL_DIR, "feature_names_input.json"), "w") as f:
        json.dump(input_feature_names, f)

    print(f"\nAll artifacts saved to {MODEL_DIR}/")
    print("  preprocessing/        -> var_selector.joblib, scaler.joblib")
    print(f"  feature_engineering/ -> corr_kept_cols.json ({len(corr_kept_cols)} cols), "
          f"mi_selected_cols.json ({len(mi_selected_cols)} cols)")
    print(f"  rf_model.joblib       -> {rf.n_estimators} trees, depth={rf.max_depth}")


if __name__ == "__main__":
    train_and_save()
