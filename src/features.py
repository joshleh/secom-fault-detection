"""
features.py — Feature selection and class imbalance handling for SECOM.

Provides correlation-based feature selection, feature importance summaries,
and utilities to handle the ~14:1 class imbalance (pass vs. fail).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

def get_feature_stats(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute summmary statistics for each features, including correlation with target.

    Returns a DataFrame index by feature naame with columns:
        mean, std, missing_pct, abs_corr_with_target
    """
    stats =  pd.DataFrame(
        {
            "mean": X.mean(),
            "std": X.std(),
            "min": X.min(),
            "max": X.max(),
            "missing_pct": X.isnull().mean() * 100,
            "corr_with_target": X.corrwith(y),
        }
    )
    stats["abs_corr_with_target"] = stats["corr_with_target"].abs()
    return stats.sort_values("abs_corr_with_target", ascending=False)

def drop_highly_correlated(
        X: pd.DataFrame, threshold: float = 0.95
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove one of each pair of features with Pearson |r| > threshold.
    
    In SECOM, many sensors are redundant (measuring the same physical process).
    Removing highly correlated features reduces multicollinearity and speeds up
    model training without losing meaningful signal.

    Parameters
    ----------
    X : pd.DataFrame — features (should already be imputed)
    treshold : float — correlation cutoff (default 0.95)

    Returns
    -------
    X_reduced : pd.DataFrame
    dropped : list[str] — names of dropped features
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X.drop(columns=to_drop)

    print(
        f"Correlation filter (|r| > {threshold}): "
        f"dropped {len(to_drop)} features, kept {X_reduced.shape[1]}"
    )
    return X_reduced, to_drop

def select_top_k_by_mutual_info(
        X: pd.DataFrame, y: pd.Series, k: int = 50, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select top-k features by mutual information with the target.
    
    Mutual information captures non-linear relationships that Pearson
    correlation misses — useful for sensor data where failure modes
    may involve threshold effects rather than linear trends.
    
    Parameters
    ----------
    X : pd.DataFrame — features (imputed, scaled)
    y : pd.Series — binary target
    k : int — number of features to keep
    random_state : int
    
    Returns
    -------
    X_selected : pd.DataFrame — top-k features
    mi_scores : pd.Series — all MI scores, sorted descending
    """
    mi = mutual_info_classif(X, y, random_state=random_state)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    top_features = mi_scores.head(k).index.tolist()
    X_selected = X[top_features]

    print(f"MI selection: kept top {k} of {X.shape[1]} features.")
    return X_selected, mi_scores

def compute_balanced_weights(y: pd.Series) -> dict:
    """
    Compute class weights inversely proportional to class frequency. 
    
    For SECOM with ~14:1 pass/fail imbalance, this gives fail samples 
    ~14 more weight — equivalent to sklearn's class_weight='balanced'.
    
    Returns
    --------
    weight_dict : dict — {0: weight_pass, 1: weight_fail}
    """
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight("balanced", classes= classes, y=y)
    weight_dict = dict(zip(classes, weights))

    print(f"Class weights: {weight_dict}")
    return weight_dict

def get_imbalance_summary(y: pd.Series) -> pd.DataFrame:
    """
    Print and return a summary of class distribution.
    
    Returns
    --------
    summary : pd.DataFrame with columns [count, pct]
    """
    counts = y.value_counts().sort_index()
    pcts = y.value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame({"count": counts, "pct": pcts.round(2)})
    summary.index = summary.index.map({0: "Pass (0)", 1: "Fail (1)"})

    print("Class Distribution Summary:")
    print(summary.to_string())
    print(f"Imbalance Ratio: {counts[0] / counts[1]:.1f} : 1")
    return summary