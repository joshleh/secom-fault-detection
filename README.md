# SECOM Fault Detection

End-to-end ML system for predicting manufacturing failures in a semiconductor
production line, from raw sensor data through to a deployed REST API.

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
| Explainability (SHAP) | 🔲 Planned |
| FastAPI Serving | 🔲 Planned |
| Docker Deployment | 🔲 Planned |

## Key Findings (EDA)
- **Class imbalance:** 93.4% pass / 6.6% fail — requires class weighting or resampling
- **Missingness:** 538 of 590 features have some missing data; 4 dropped (>50%), remainder median-imputed
- **Dimensionality:** 140 zero-variance features dropped, leaving 446 features for modeling
- **Correlated sensors:** 7 highly correlated pairs (|r| > 0.9), candidates for PCA or deduplication

## Model Results
Three approaches compared, all logged to MLflow:

| Model | Type | ROC-AUC | Fail Recall | Fail F1 |
|---|---|---|---|---|
| Isolation Forest | Unsupervised | 0.57 | 19% | 0.20 |
| **Random Forest** | Supervised | **0.79** | **33%** | **0.34** |
| LSTM Autoencoder | Autoencoder | 0.58 | 10% | 0.11 |

**Random Forest** is the strongest model after threshold tuning (0.50 → 0.20)
to address the 14:1 class imbalance. The LSTM Autoencoder's weaker performance
is expected — SECOM features lack true sequential structure, and 1,170
pass-only training samples is small for deep learning. Top predictive sensors
include sensor_103, sensor_59, and sensor_33.

## Approach
- **Preprocessing:** Median imputation, zero-variance removal, standard scaling
- **Models:** Isolation Forest (unsupervised baseline), Random Forest with class weighting (supervised baseline), LSTM Autoencoder trained on pass-only data (deep learning)
- **Imbalance handling:** Class weighting + decision threshold tuning
- **Tracking:** MLflow experiment tracking with params, metrics, confusion matrices, and serialized models
- **Evaluation:** PR-AUC and recall prioritized — missing a failure costs more than a false alarm
- **Explainability:** SHAP feature importance (next)
- **Serving:** FastAPI REST endpoint + Docker (planned)

## Repo Structure
```
├── api/                        # FastAPI app (planned)
├── notebooks/
│   ├── 01_eda.ipynb            # Data loading & exploratory analysis
│   ├── 02_modeling.ipynb       # Feature engineering & baseline models
│   └── 03_model_comparison.ipynb  # Multi-model comparison + MLflow
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── data/                       # gitignored — auto-downloaded by notebook
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb   # downloads data automatically
```
