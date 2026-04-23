# Model Card — SECOM Yield Fault Detection

> A short, opinionated model card following the [Model Cards for Model
> Reporting](https://arxiv.org/abs/1810.03993) framework, scoped to the
> needs of an internal yield-debug tool.

## 1. Model details

| Field | Value |
|---|---|
| Model name | `secom-rf-classifier` |
| Version | `1.0.0` |
| Type | Calibrated-imbalance binary classifier (Random Forest) |
| Library | scikit-learn 1.5.2 |
| Owner | [@joshleh](https://github.com/joshleh) |
| Last trained | committed snapshot in `dashboard_assets/models/` |
| Inputs | 446 numeric sensor readings per wafer |
| Outputs | `prediction ∈ {PASS, FAIL}` and `probability ∈ [0, 1]` |
| Model file | `rf_model.joblib` (~1.3 MB) |

## 2. Intended use

**Primary use.** Internal triage tool for engineers investigating
specific wafer failures. The dashboard pairs every prediction with a
SHAP explanation so the engineer can see *which sensors* contributed
most, then jump to the Compare / Cluster / Drift pages to look for
recurring patterns.

**Out-of-scope uses.**
- Real-time inline disposition (this model is not certified for
  automated wafer scrapping decisions).
- Datasets from other fabs / different sensor suites — no transfer
  guarantees.
- Anything where false positives carry unrecoverable cost; the tuned
  threshold prioritizes recall.

## 3. Training data

- **Dataset:** [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom),
  1,567 wafers × 590 anonymized sensors collected from a real
  semiconductor manufacturing line.
- **Class balance:** ~6.6 % fail (104 of 1,567).
- **Reproducibility:** `scripts/fetch_data.py` pins the upstream files
  by SHA-256:
  - `secom.data` → `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`
  - `secom_labels.data` → `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03`
- **Cleaning:** dropped 4 sensors with > 50 % missingness, median-imputed
  the rest → 446 features (saved as `X_clean.csv`).

## 4. Training procedure

`src/train.py` reproduces the model end-to-end:

1. Variance filter (drop near-constant sensors)
2. `StandardScaler`
3. Pearson correlation filter at \|r\| > 0.95 (kept 272 of 446)
4. Mutual-information selection: top **50** features
5. **Stratified 5-fold CV** (mean ± std reported)
6. `RandomForestClassifier(n_estimators=200, max_depth=10,
   min_samples_leaf=5, class_weight="balanced", random_state=42)` on
   80% train / 20% validation stratified split
7. **F1-optimal decision threshold** tuned on the validation fold and
   persisted to `models/threshold.json`

A `calibrate()` helper (isotonic) is available in `src/models.py` for
opt-in use from notebooks but is not applied in the default pipeline:
SHAP's `TreeExplainer` requires the bare RF, and with ~83 fail samples
in train, isotonic calibration is unreliable.

## 5. Evaluation

### Stratified 5-fold cross-validation (held-out folds)

| Metric | Mean | Std |
|---|---|---|
| ROC-AUC | 0.747 | ±0.044 |
| PR-AUC  | 0.215 | ±0.034 |
| Fail-F1 | 0.070 | ±0.065 |

> The very low Fail-F1 across folds reflects that with default 0.5
> threshold the model rarely commits to FAIL. That is exactly why the
> threshold is tuned on the held-out validation fold.

### Held-out 20 % validation set @ tuned threshold (0.222)

| Class | Precision | Recall | F1   | Support |
|---|---|---|---|---|
| Pass  | 0.97      | 0.84   | 0.90 | 293     |
| Fail  | 0.23      | 0.67   | 0.35 | 21      |
| **Overall ROC-AUC** | | | **0.804** | |

Threshold tuning **doubles fail recall (33 % → 67 %)** vs the default
0.5 threshold, at the cost of increased false positives — the right
trade-off for a triage tool where missing a defect is worse than
flagging an extra wafer for review.

## 6. Limitations & ethical considerations

- **Sensor names are anonymized.** Predictions and SHAP explanations
  are correct mathematically but the natural-language summaries
  reference `sensor_NN` rather than physical components.
- **Imbalance.** With only 104 failures in the entire dataset, every
  metric has wide confidence intervals (see CV std above).
- **Distributional drift.** A SECOM-trained model should not be
  deployed against a *different* fab without retraining and re-running
  the drift report (`/drift` endpoint or dashboard tab) on the new
  population.
- **No fairness analysis.** SECOM is a manufacturing dataset with no
  human subjects, so demographic fairness is not applicable.

## 7. How to reproduce

```bash
make install-dev
make fetch-data        # downloads + hash-verifies the UCI files
python notebooks/01_eda.ipynb   # writes data/processed/X_clean.csv
make train             # rebuilds models/ from scratch
make snapshot          # syncs models/ -> dashboard_assets/
make test              # 52 tests
```

## 8. Changelog

- **1.0.0** — Initial release: 50-feature RF + SHAP, F1-tuned threshold
  (0.222), drift baseline persisted, 5-fold CV reported, batch +
  metadata + drift endpoints.
