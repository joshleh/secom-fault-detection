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
| Feature Engineering | 🔲 Planned |
| Modeling + MLflow Tracking | 🔲 Planned |
| Explainability (SHAP) | 🔲 Planned |
| FastAPI Serving | 🔲 Planned |
| Docker Deployment | 🔲 Planned |

## Key Findings (EDA)
- **Class imbalance:** 93.4% pass / 6.6% fail — requires class weighting or resampling
- **Missingness:** 538 of 590 features have some missing data; 4 dropped (>50%), remainder median-imputed
- **Dimensionality:** 140 zero-variance features dropped, leaving 446 features for modeling
- **Correlated sensors:** 7 highly correlated pairs (|r| > 0.9), candidates for PCA or deduplication

## Planned Approach
- **Models:** Logistic Regression + Random Forest baseline, XGBoost
- **Imbalance:** SMOTE or class weighting
- **Tracking:** MLflow experiment tracking for all runs
- **Evaluation:** F2-score and recall prioritized — missing a failure costs more than a false alarm
- **Explainability:** SHAP feature importance to identify which sensors drive predictions
- **Serving:** FastAPI REST endpoint + Dockerized for deployment

## Repo Structure
```
├── api/
│ ├── main.py # FastAPI app
│ └── schema.py # Request/response schemas
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_modeling.ipynb
│ └── 03_explainability.ipynb
├── src/
│ ├── preprocess.py
│ ├── features.py
│ ├── train.py
│ └── evaluate.py
├── mlflow/ # Experiment tracking
├── data/ # gitignored — auto-downloaded by notebook
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb # downloads data automatically
```