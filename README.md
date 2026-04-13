# Semiconductor Yield Fault Detection

ML-driven diagnostic tool for investigating semiconductor manufacturing failures.
Combines a trained fault-detection model with interactive root-cause analysis,
sensor-level deviation tracking, and failure pattern clustering.

## Background

Semiconductors (chips) are manufactured on thin silicon discs called **wafers**. Each wafer passes through dozens of manufacturing steps — etching, deposition, lithography, etc. — monitored by hundreds of **sensors** that record temperatures, pressures, gas flows, and other measurements.

At the end of the line, each wafer is tested: **pass** (good) or **fail** (defective). When failures spike, engineers need to figure out *which sensors* (and therefore which manufacturing steps) are responsible. That investigation is called **yield debugging** — "yield" being the percentage of wafers that pass.

This project automates the first stage of that investigation: given a wafer's sensor readings, predict whether it will pass or fail, explain which sensors contributed most, and flag readings that look abnormal compared to healthy wafers.

**Dataset:** [UCI SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM)
— 1,567 wafers × 590 sensor readings, with pass/fail labels

## What This Does

When investigating manufacturing failures, an engineer needs to answer three questions:

1. **Which wafers are high-risk?** → The model assigns a failure probability to every wafer
2. **What sensors are driving the failure?** → SHAP explanations rank the most impactful sensors (SHAP is a method that breaks down a prediction into per-feature contributions)
3. **Is this a known failure pattern?** → Baseline deviation analysis compares each wafer against the "normal" population (wafers that passed), and failure clustering groups similar failures together

This project provides all three through a **Streamlit dashboard** for interactive debugging
and a **Jupyter notebook** for deeper batch analysis.

## Dashboard

![Dashboard Screenshot](docs/dashboard_screenshot.png?v=2)

The dashboard lets you select any wafer and instantly see:

- **Diagnostic Overview** — pass/fail prediction with the model's confidence
- **Top Sensor Drivers** — which sensors influenced the prediction most (using SHAP, a model explainability method)
- **Deviation from Healthy Baseline** — how far each sensor reading is from the normal range (measured in standard deviations, a.k.a. "z-scores")
- **Sample vs Baseline Comparison** — side-by-side visual of this wafer vs the average passing wafer
- **Failure Pattern Summary** — a plain-English diagnostic summary generated from the data (not an LLM)

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

### EDA (Exploratory Data Analysis)
- **Class imbalance:** 93.4% of wafers pass, only 6.6% fail — the model needs special handling to avoid ignoring the rare failures
- **Missing data:** 538 of 590 sensors have gaps; 4 sensors are dropped entirely (>50% missing), the rest are filled with the median value
- **Constant sensors:** 140 sensors never change — they carry no useful information and are removed
- **Redundant sensors:** 174 sensors are near-duplicates of others (correlation > 0.95) and are removed to avoid noise

### Feature Engineering Pipeline

The raw data goes through a reduction pipeline before the model sees it:

```
590 raw sensor readings
 → Drop >50% missing (4 removed)                    586
 → Drop constant sensors (140 removed)              446  ← saved as X_clean.csv
 → Remove near-zero-variance + standardize           446
 → Remove highly correlated duplicates (|r| > 0.95)  272
 → Select top 50 by mutual information                50  ← what the model actually uses
```

"Mutual information" measures how much knowing a sensor's value tells you about pass/fail — it captures non-linear relationships that simple correlation misses.

This pipeline is implemented in `src/preprocess.py` and `src/features.py`, and shared across
training (`src/train.py`), the API (`api/main.py`), and the dashboard (`src/diagnostics.py`).

### Model Comparison

Three approaches were compared, all tracked with [MLflow](https://mlflow.org/) (an experiment tracking tool):

| Model | Type | ROC-AUC | Fail Recall | Fail F1 |
|---|---|---|---|---|
| Isolation Forest | Anomaly detection | 0.57 | 19% | 0.20 |
| **Random Forest** | Classification | **0.80** | **33%** | **0.34** |
| LSTM Autoencoder | Deep learning | 0.58 | 10% | 0.11 |

**What the metrics mean:**
- **ROC-AUC** (0-1): overall ranking quality — how well the model separates pass from fail
- **Fail Recall**: of all actual failures, what percentage did the model catch?
- **Fail F1**: balance between catching failures and not raising too many false alarms

Random Forest was selected because it had the best overall performance. It uses class weighting to pay extra attention to the rare failure cases — missing a defective wafer is more costly than a false alarm in manufacturing.

### Explainability

SHAP (SHapley Additive exPlanations) breaks each prediction into per-sensor contributions, answering "why did the model predict FAIL for this wafer?" A small subset of sensors drives most predictions — **sensor_103** and **sensor_59** have the highest average impact. Their behavior shows non-linear threshold effects: the sensor reading is fine until it crosses a certain value, then failure risk jumps sharply.

## API

A [FastAPI](https://fastapi.tiangolo.com/) endpoint provides the same prediction + explanation as the dashboard, but as a REST API (useful for integrating with other systems):

```
POST /predict
{
    "features": [0.23, -1.1, 0.88, ...]   // 446 sensor readings for one wafer
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
│   └── 05_yield_debug_analysis.ipynb # Failure investigation & pattern clustering
├── src/
│   ├── preprocess.py                 # Variance filter, scaling, imputation
│   ├── features.py                   # Correlation filter, MI selection
│   ├── train.py                      # Train RF + save pipeline artifacts
│   └── diagnostics.py                # "Normal" baseline analysis, deviation
│                                     #   measurement, diagnostic summary generation
├── dashboard_assets/                 # Committed snapshot for zero-setup demos
│   ├── data/                         #   X_clean.csv, y.csv (~4.8 MB)
│   └── models/                       #   rf_model.joblib + pipeline artifacts (~1.3 MB)
├── tests/
│   └── test_api.py                   # API smoke tests
├── data/                             # gitignored — generated by notebooks
├── models/                           # gitignored — generated by train.py
├── Dockerfile
├── requirements.txt                  # Streamlit Cloud / dashboard runtime deps
├── requirements-dev.txt              # Full dev deps (notebooks, training, API)
└── README.md
```

### About `dashboard_assets/`

The `data/` and `models/` directories are gitignored because they are generated
by running the notebooks and `src/train.py`. However, the Streamlit dashboard
needs that processed data and a trained model to display anything.

To make the dashboard work **immediately after cloning** (no notebooks, no training
required), a snapshot of the necessary files is committed in `dashboard_assets/`.
The dashboard checks this directory first, and falls back to `data/` + `models/` if
it's missing. This keeps the full pipeline untouched while making the demo
self-contained (~6 MB total).

## Dependencies

This project uses two requirements files:

| File | What it covers | When to use |
|---|---|---|
| `requirements.txt` | Dashboard runtime only (numpy, pandas, scikit-learn, joblib, plotly, shap, streamlit) | Streamlit Cloud deploys from this automatically. Also sufficient if you only want to run the dashboard locally. |
| `requirements-dev.txt` | Everything above **plus** torch, mlflow, fastapi, uvicorn, jupyter, matplotlib, seaborn | Local development — running notebooks, training models, serving the API. |

`requirements-dev.txt` inherits from `requirements.txt` via `-r requirements.txt`,
so you only ever need to install one of them.

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
pip install -r requirements-dev.txt
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
- **Preprocessing:** Fill missing values with medians, remove constant sensors, standardize scales
- **Feature Selection:** Remove near-duplicate sensors, then select the top 50 most informative sensors using mutual information
- **Models Compared:** Isolation Forest (anomaly detection), Random Forest (classification, deployed), LSTM Autoencoder (deep learning)
- **Handling Rare Failures:** Class weighting gives extra importance to the rare fail cases during training
- **Experiment Tracking:** MLflow logs every model's parameters, metrics, and artifacts for reproducibility
- **Explainability:** SHAP breaks each prediction into per-sensor contributions so you can see *why* the model predicted fail
- **Diagnostics:** Baseline deviation analysis (how far from normal?), failure clustering (are failures similar to each other?), and auto-generated plain-English summaries
- **Serving:** FastAPI REST endpoint with SHAP explanations, Dockerized for deployment
- **Dashboard:** Streamlit interactive tool for inspecting individual wafers

## Future Work
- Fine-tune the decision threshold to optimize the tradeoff between catching failures and false alarms
- Monitor for data drift — detect when incoming sensor data stops matching what the model was trained on
- Integration with manufacturing execution systems (MES) for real-time tracking
- Build a library of known failure patterns from past investigations
