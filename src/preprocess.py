"""
preprocess.py — Reusable preprocessing pipeline for SECOM semiconductor data.

Functions handle missing value imputation, low-variance feature removal,
and feature scaling. Designed to be imported by notebooks and the FastAPI
serving layer so the same transforms are used in training and inference.

Data conventions (matching 01_eda.ipynb):
- Raw Labels: 1 = pass, -1 = fail (UCI convention)
- Binary Labels: 0 = pass, 1 = fail (our convention set in EDA)
- Column Names: sensor_0...sensor_589
- EDA already saves X_clean.csv (446 features, median-imputed, no >50%-missing cols)
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ==========================================================================
# Data Loading
# ==========================================================================

def load_raw(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw SECOM data from CSV files.

    Returns:
    --------
    X : pd.DataFrame (1567, 590) - raw sensor features with NaN
    Y: pd.Series (1567,) - binary labels (0=pass, 1=fail)
    """
    data_path = Path(data_dir)

    # Features: space-separated, no header
    X = pd.read_csv(data_path / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    # Labels: col 0 = label, col 1 = timestamp
    labels_df = pd.read_csv(
        data_path / "secom_labels.data", sep=r"\s+", header=None, 
        names=["label", "timestamp"]
    )

    # Raw: 1 = pass, -1 = fail --> Map to Binary: 0 = pass, 1 = fail
    y = labels_df["label"].map({1: 0, -1: 1})  # Convert to binary labels

    return X, y

def load_clean(data_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the already-cleaned saved by 01_eda.ipynb.
    
    X_clean.csv: 446 features, >50% missing & zero variance columns dropped,
                median imputed (no NaN remaining)
    y.csv: binary labels (0=pass, 1=fail)
    """

    data_path = Path(data_dir)
    X = pd.read_csv(data_path / "X_clean.csv")
    y = pd.read_csv(data_path / "y.csv")["label"]  # Load as Series
    return X, y

# ==========================================================================
# Cleaning steps (mirrors EDA logic as reusable functions)
# ==========================================================================

def drop_high_missing(
    X: pd.DataFrame, threshold: float = 0.5
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop features with missing fraction above `threshold`.

    EDA finding: 4 features have >50% missing values.
    """
    miss_frac = X.isnull().mean()
    to_drop = miss_frac[miss_frac > threshold].index.tolist()
    X_out = X.drop(columns=to_drop)
    logger.info(
        "Dropped %d features > %.0f%% missing. Remaining: %d",
        len(to_drop), threshold * 100, X_out.shape[1],
    )
    return X_out, to_drop

def drop_zero_variance(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop features with exactly zero standard deviation."""
    zero_var = X.columns[X.std() == 0].tolist()
    X_out = X.drop(columns=zero_var)
    logger.info(
        "Dropped %d zero-variance features. Remaining: %d",
        len(zero_var), X_out.shape[1],
    )
    return X_out, zero_var

def impute_missing(
        X: pd.DataFrame,
        strategy: str = "median", 
        fit: bool = True,
        imputer: SimpleImputer | None = None
    ) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing values using median (default).
    
    Median is preferred over KNN for SECOM because:
    - 538/590 features have missing values — KNN neighbor distances
    are unreliable in such high-dimensional, sparse feature spaces
    - Sensor distributions are heavy-tailed with outliers — median is
    more robust than mean
    - KNN imputation on 590 features is computationally expensive
    with marginal quality gains
    - Median is the industry standard for tabular sensor data
    """
    if fit:
        imputer = SimpleImputer(strategy=strategy)
        X_out = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns, 
            index=X.index
        )
    else:
        if imputer is None:
            raise ValueError("Must provide fitted imputer when fit=False")
        X_out = pd.DataFrame(
            imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
        
    return X_out, imputer

def remove_low_variance(
        X: pd.DataFrame,
        threshold: float = 0.0,
        fit: bool = True,
        selector: VarianceThreshold | None = None
    ) -> tuple[pd.DataFrame, VarianceThreshold]:
    """
    Drop features with variance at or below `threshold`.

    threshold=0.0 removes only constant features (safest default).
    After scaling, ~0.01 can catch near-constant features too.
    """
    if fit:
        selector = VarianceThreshold(threshold=threshold)
        X_out = pd.DataFrame(
            selector.fit_transform(X), 
            columns=X.columns[selector.get_support()],
            index=X.index
        )
    else:
        if selector is None:
            raise ValueError("Must provide fitted selector when fit=False")
        X_out = pd.DataFrame(
            selector.transform(X), 
            columns=X.columns[selector.get_support()], 
            index=X.index
        )
    
    n_dropped = X.shape[1] - X_out.shape[1]
    logger.info("Variance filter: dropped %d features, kept %d", n_dropped, X_out.shape[1])
    return X_out, selector

def scale_features(
        X: pd.DataFrame,
        fit: bool = True,
        scaler: StandardScaler | None = None,
    ) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features to zero mean and unit variance.

    StandardScaler is appropriate here because:
    - SECOM features have very different scales (found in EDA)
    - Logistic Regression and distance-based methods need scaled inputs
    - Tree models are scale-invariant but scaling doesn't hurt them
    """
    if fit:
        scaler = StandardScaler()
        X_out = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
            )
    else:
        if scaler is None:
            raise ValueError("Must provide fitted scaler when fit=False")
        X_out = pd.DataFrame(
            scaler.transform(X), columns=X.columns, index=X.index
        )
    return X_out, scaler

# ==========================================================================
# Full pipeline
# ==========================================================================

def run_preprocessing_pipeline(
        X: pd.DataFrame,
        variance_threshold: float = 0.0,
        fit: bool = True,
        artifacts: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline on already-cleaned data.
    
    Expected input: X_clean from EDA (446 features, already imputed).
    Pipeline: variance filter -> scale.
    
    For raw data, call drop_high_missing() and impute_missing() first.

    Parameters:
    -----------
    X: pd.DataFrame — cleaned features (from load_clean or EDA output)
    variance_threshold: float — minimum variance to keep a feature
    fit: bool — True for training, False for inference
    artifacts: dict | None — fitted transformers (required if fit=False)

    Returns:
    --------
    X_processed: pd.DataFrame
    artifacts: dict — {'var_selector', 'scaler'} fitted objects
    """

    if not fit and artifacts is None:
        raise ValueError("Must provide artifacts dict when fit=False (inference mode)")

    if fit:
        artifacts = {}
    
    # Step 1: Low-variance removal (may catch additional near-constant features)
    X_var, artifacts['var_selector'] = remove_low_variance(
        X, threshold=variance_threshold, fit=fit, selector=artifacts.get('var_selector')
    )

    # Step 2: Scaling
    X_scaled, artifacts['scaler'] = scale_features(
        X_var, fit=fit, scaler=artifacts.get('scaler')
    )

    return X_scaled, artifacts

# ==========================================================================
# Artifact persistence
# ==========================================================================

def save_artifacts(artifacts: dict, save_dir: str = "models/preprocessing") -> None:
    """Save fitted preprocessing objects for later inference."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        joblib.dump(obj, save_path / f"{name}.joblib")
    logger.info("Saved %d preprocessing artifacts to %s", len(artifacts), save_path)

def load_artifacts(save_dir: str = "models/preprocessing") -> dict:
    """Load fitted preprocessing objects."""
    save_path = Path(save_dir)
    artifacts = {}
    for fpath in save_path.glob("*.joblib"):
        artifacts[fpath.stem] = joblib.load(fpath)
    logger.info("Loaded %d preprocessing artifacts from %s", len(artifacts), save_path)
    return artifacts