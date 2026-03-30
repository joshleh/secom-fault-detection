# SECOM Fault Detection

End-to-end ML system for predicting manufacturing failures in a semiconductor
production line, from raw sensor data through to a deployed REST API with
SHAP-based explanations.

**Dataset:** [UCI SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM)
— 1,567 samples × 590 features, binary pass/fail yield labels

## Motivation
In semiconductor manufacturing, catching a failing wafer early saves
significant cost. This project builds a binary classifier to flag likely
failures from real-time sensor readings, with a focus on the challenges
that make this hard in practice: extreme class imbalance (14:1),
high missingness, and correlated sensors.

## Project Status
| Phase | Status |
|---|---|
| EDA | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Modeling + MLflow Tracking | ✅ Complete |
| Explainability (SHAP) | ✅ Complete |
| FastAPI Serving | ✅ Complete |
| Docker Deployment | ✅ Complete |

## Key Findings (EDA)
- **Class imbalance:** 93.4% pass / 6.6% fail — requires class weighting or resampling
- **Missingness:** 538 of 590 features have some missing data; 4 dropped (>50%), remainder median-imputed
- **Dimensionality:** 140 zero-variance features dropped, leaving 446 features for modeling
- **Correlated sensors:** 174 features removed at |r| > 0.95, then top 50 selected by mutual information

## Feature Engineering Pipeline
Raw sensor data goes through a multi-stage pipeline before reaching the model:

```
590 raw features
 → Drop >50% missing (4 removed)         586
 → Drop zero-variance (140 removed)      446  ← X_clean.csv
 → Variance filter + StandardScaler       446
 → Correlation filter |r| > 0.95          272
 → Mutual information top-50               50  ← model input
```

This pipeline is implemented in `src/preprocess.py` and `src/features.py`,
shared between training (`src/train.py`) and serving (`api/main.py`) to
guarantee identical transforms at inference time.

## Model Results
Three approaches compared, all logged to MLflow:

| Model | Type | ROC-AUC | Fail Recall | Fail F1 |
|---|---|---|---|---|
| Isolation Forest | Unsupervised | 0.57 | 19% | 0.20 |
| **Random Forest** | Supervised | **0.80** | **33%** | **0.34** |
| LSTM Autoencoder | Autoencoder | 0.58 | 10% | 0.11 |

**Random Forest** is the strongest model after threshold tuning (0.50 → 0.20)
to address the 14:1 class imbalance. The LSTM Autoencoder's weaker performance
is expected — SECOM features lack true sequential structure, and 1,170
pass-only training samples is small for deep learning.

## Explainability (SHAP)
SHAP TreeExplainer on the Random Forest reveals that a small subset of sensors
drives most of the model's fault predictions — **sensor_103** and **sensor_59**
have the highest mean |SHAP| values. Dependence plots show non-linear threshold
effects (sensor readings are benign until a critical value, then failure risk
spikes), and interaction coloring surfaces coupled sensor pairs that likely map
to the same process step. Full beeswarm, dependence, and waterfall plots are in
`notebooks/04_explainability.ipynb`.

## API
The FastAPI endpoint reproduces the full feature engineering pipeline at
inference time and returns SHAP-based explanations with every prediction:

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

Interpretability is baked into the API response, not just a notebook
afterthought — a downstream system or engineer can see *which sensors*
drove each prediction.

## Repo Structure
```
├── api/
│   └── main.py                 # FastAPI endpoint with SHAP explanations
├── notebooks/
│   ├── 01_eda.ipynb            # Data loading & exploratory analysis
│   ├── 02_modeling.ipynb       # Feature engineering & baseline models
│   ├── 03_model_comparison.ipynb  # Multi-model comparison + MLflow
│   └── 04_explainability.ipynb    # SHAP analysis + stakeholder interpretation
├── src/
│   ├── preprocess.py           # Variance filter, scaling, imputation
│   ├── features.py             # Correlation filter, MI selection
│   ├── train.py                # Train RF + save all pipeline artifacts
│   └── evaluate.py             # Evaluation utilities
├── tests/
│   └── test_api.py             # API smoke tests
├── models/                     # Saved artifacts (created by train.py)
│   ├── preprocessing/          # var_selector.joblib, scaler.joblib
│   ├── feature_engineering/    # corr_kept_cols.json, mi_selected_cols.json
│   └── rf_model.joblib
├── data/                       # gitignored
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download data and run EDA:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**3. Train the model** (saves all pipeline artifacts to `models/`):
```bash
python src/train.py
```

**4. Start the API:**
```bash
uvicorn api.main:app --reload --port 8000
```

**5. Test it:**
```bash
python tests/test_api.py
```

**Docker:**
```bash
docker build -t secom-api .
docker run -p 8000:8000 secom-api
```

## Approach
- **Preprocessing:** Median imputation, zero-variance removal, standard scaling (`src/preprocess.py`)
- **Feature Engineering:** Correlation deduplication (|r| > 0.95), mutual information selection (top 50) (`src/features.py`)
- **Models:** Isolation Forest (unsupervised baseline), Random Forest with class weighting (supervised), LSTM Autoencoder trained on pass-only data (deep learning)
- **Imbalance handling:** Class weighting + decision threshold tuning
- **Tracking:** MLflow experiment tracking with params, metrics, confusion matrices, and serialized models
- **Evaluation:** PR-AUC and recall prioritized — missing a failure costs more than a false alarm
- **Explainability:** SHAP TreeExplainer — beeswarm, dependence, and waterfall plots
- **Serving:** FastAPI with SHAP explanations per prediction, Dockerized for deployment
