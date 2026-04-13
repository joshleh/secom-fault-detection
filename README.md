# Semiconductor Yield Debug Dashboard

ML-driven diagnostic tool for investigating semiconductor manufacturing failures.
Combines a trained fault-detection model with interactive root-cause analysis,
sensor-level deviation tracking, and failure pattern clustering — built for
engineers working on yield improvement and silicon debug.

**Dataset:** [UCI SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM)
— 1,567 wafer samples × 590 sensor features, binary pass/fail yield labels

## What This Does

A yield engineer investigating a failing lot needs to answer three questions:

1. **Which wafers are high-risk?** → The model assigns failure probabilities to every sample
2. **What sensors are driving the failure?** → SHAP explanations rank the most impactful sensors
3. **Is this a known failure pattern?** → Baseline deviation analysis and failure clustering
   reveal whether the issue is systematic or novel

This project provides all three through a Streamlit dashboard for interactive debugging
and a Jupyter notebook for deeper batch analysis.

## Dashboard

The Yield Debug Dashboard provides per-sample diagnostics:

- **Sample Diagnostic Overview** — pass/fail prediction with confidence score
- **Top Sensor Drivers** — SHAP-based ranking of contributing sensors
- **Deviation from Healthy Baseline** — z-score comparison against pass-only population
- **Sample vs Baseline Comparison** — visual overlay of sample values vs normal range
- **Preliminary Failure Pattern Summary** — rule-based root-cause narrative grounded in actual sensor data

```bash
streamlit run app/streamlit_app.py
```

## Project Status
| Phase | Status |
|---|---|
| EDA & Data Cleaning | ✅ Complete |
| Feature Engineering Pipeline | ✅ Complete |
| Model Benchmarking + MLflow | ✅ Complete |
| SHAP Explainability | ✅ Complete |
| FastAPI Inference Service | ✅ Complete |
| Yield Debug Dashboard (Streamlit) | ✅ Complete |
| Failure Pattern Analysis Notebook | ✅ Complete |
| Docker Deployment | ✅ Complete |

## Key Findings

### EDA
- **Class imbalance:** 93.4% pass / 6.6% fail — requires class weighting
- **Missingness:** 538 of 590 features have missing data; 4 dropped (>50%), remainder median-imputed
- **Dimensionality:** 140 zero-variance features dropped, leaving 446 for modeling
- **Correlated sensors:** 174 features removed at |r| > 0.95, then top 50 selected by mutual information

### Feature Engineering Pipeline
```
590 raw features
 → Drop >50% missing (4 removed)         586
 → Drop zero-variance (140 removed)      446  ← X_clean.csv
 → Variance filter + StandardScaler       446
 → Correlation filter |r| > 0.95          272
 → Mutual information top-50               50  ← model input
```

Implemented in `src/preprocess.py` and `src/features.py`, shared between
training (`src/train.py`), serving (`api/main.py`), and diagnostics (`src/diagnostics.py`).

### Model Comparison
Three approaches compared, all logged to MLflow:

| Model | Type | ROC-AUC | Fail Recall | Fail F1 |
|---|---|---|---|---|
| Isolation Forest | Unsupervised | 0.57 | 19% | 0.20 |
| **Random Forest** | Supervised | **0.80** | **33%** | **0.34** |
| LSTM Autoencoder | Autoencoder | 0.58 | 10% | 0.11 |

Random Forest selected for deployment. The model uses `class_weight="balanced"`
and threshold tuning to prioritize failure recall over precision — missing a
failing wafer is more costly than a false alarm.

### Explainability
SHAP TreeExplainer reveals that a small subset of sensors drives most fault
predictions — **sensor_103** and **sensor_59** have the highest mean |SHAP|
values. Dependence plots show non-linear threshold effects consistent with
physical process limits.

## API

The FastAPI service reproduces the full feature pipeline at inference time:

```
POST /predict
{
    "features": [0.23, -1.1, 0.88, ...]   # 446 sensor readings
}

→ returns:
{
    "prediction": "FAIL",
    "probability": 0.87,
    "top_contributing_features": [
        {"feature": "sensor_59",  "shap_value": 0.142},
        {"feature": "sensor_103", "shap_value": 0.098},
        ...
    ]
}
```

```bash
uvicorn api.main:app --reload --port 8000
```

## Repo Structure
```
├── app/
│   └── streamlit_app.py              # Yield Debug Dashboard
├── api/
│   └── main.py                       # FastAPI inference endpoint + SHAP
├── notebooks/
│   ├── 01_eda.ipynb                  # Data loading & exploratory analysis
│   ├── 02_modeling.ipynb             # Feature engineering & baseline models
│   ├── 03_model_comparison.ipynb     # Multi-model comparison + MLflow
│   ├── 04_explainability.ipynb       # SHAP analysis (Random Forest)
│   └── 05_yield_debug_analysis.ipynb # Failure investigation & clustering
├── src/
│   ├── preprocess.py                 # Variance filter, scaling, imputation
│   ├── features.py                   # Correlation filter, MI selection
│   ├── train.py                      # Train RF + save pipeline artifacts
│   └── diagnostics.py                # Baseline analysis, deviation tracking,
│                                     #   root-cause summary generation
├── dashboard_assets/                 # Committed snapshot for zero-setup demos
│   ├── data/                         #   X_clean.csv, y.csv (~4.8 MB)
│   └── models/                       #   rf_model.joblib + pipeline artifacts (~1.3 MB)
├── tests/
│   └── test_api.py                   # API smoke tests
├── data/                             # gitignored — generated by notebooks
├── models/                           # gitignored — generated by train.py
├── Dockerfile
├── requirements.txt
└── README.md
```

### About `dashboard_assets/`

The `data/` and `models/` directories are gitignored because they are generated
at runtime by the notebooks and `src/train.py`. However, the Streamlit dashboard
needs processed data and a trained model to run.

To make the dashboard work immediately after cloning (no notebooks, no training),
a snapshot of the processed data and trained model is committed in `dashboard_assets/`.
The dashboard checks this directory first and falls back to `data/` + `models/` if
it doesn't exist. This keeps the full training pipeline untouched while making the
demo self-contained (~6 MB total).

## Setup

### Quick Start (dashboard only)

The dashboard works immediately after cloning — no data download or training needed.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Full Pipeline (EDA → training → dashboard)

To reproduce the full workflow from scratch:

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download data and run EDA** (generates `data/processed/`):
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**3. Train the model** (generates `models/`):
```bash
python src/train.py
```

**4. Launch the dashboard:**
```bash
streamlit run app/streamlit_app.py
```

**5. (Optional) Start the API:**
```bash
uvicorn api.main:app --reload --port 8000
```

**6. (Optional) Run the yield debug analysis notebook:**
```bash
jupyter notebook notebooks/05_yield_debug_analysis.ipynb
```

## Technical Approach
- **Preprocessing:** Median imputation, zero-variance removal, standard scaling
- **Feature Engineering:** Correlation deduplication (|r| > 0.95), mutual information selection (top 50)
- **Models:** Isolation Forest (unsupervised baseline), Random Forest with class weighting (deployed), LSTM Autoencoder (deep learning baseline)
- **Imbalance handling:** Class weighting + decision threshold tuning
- **Tracking:** MLflow experiment tracking with params, metrics, and serialized models
- **Explainability:** SHAP TreeExplainer with per-prediction feature contributions
- **Diagnostics:** Baseline deviation analysis, failure clustering, rule-based root-cause summaries
- **Serving:** FastAPI with SHAP explanations per prediction, Dockerized for deployment
- **Dashboard:** Streamlit interactive diagnostic tool with sensor-level investigation

## Future Work
- Threshold tuning with PR-curve optimization for production alert sensitivity
- Data drift monitoring on incoming sensor streams
- Integration with MES/FDC systems for real-time yield tracking
- Expanded failure mode library from historical debug investigations
