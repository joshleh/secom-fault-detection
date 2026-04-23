# Semiconductor Yield Fault Detection

[![CI](https://github.com/joshleh/secom-fault-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/joshleh/secom-fault-detection/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](#dependencies)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20demo-FF4B4B?logo=streamlit&logoColor=white)](#dashboard)

End-to-end **machine-learning system** for catching defective semiconductor wafers
before they ship — and explaining *why* the model thinks they're defective. Trained
on the [UCI SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM) sensor dataset
(1,567 wafers × 590 sensors, 6.6% failure rate). Ships with an interactive
**Streamlit dashboard**, a **FastAPI** inference service, **Docker** deployment,
**drift monitoring**, **CI**, and a documented **model card**.

> **TL;DR for reviewers** — Random Forest with class weighting and an F1-tuned
> decision threshold. ROC-AUC **0.747 ± 0.04** (5-fold stratified CV), Fail-Recall
> **67%** at the tuned threshold (vs. 33% at default 0.5). Every prediction is
> explained with SHAP, every endpoint is tested in CI, and the trained model
> is bundled into a 100 MB Docker image that boots in under 5 s.

---

## Table of contents

- [Why this project](#why-this-project)
- [Dashboard tour](#dashboard) — five pages, with screenshots
- [Architecture](#architecture)
- [Results](#results)
- [Tech stack](#tech-stack)
- [Repo structure](#repo-structure)
- [Setup](#setup)
- [API](#api)
- [Drift monitoring](#drift-monitoring)
- [Engineering quality](#engineering-quality)
- [Model card & limitations](#model-card)
- [Future work](#future-work)

---

## Why this project

Semiconductors are manufactured on thin silicon discs (**wafers**) that pass
through dozens of fabrication steps — etching, deposition, lithography, etc. —
each monitored by hundreds of sensors recording temperature, pressure, gas flow,
and so on. At the end of the line every wafer is tested: **pass** (good) or
**fail** (defective). When failures spike, process engineers have to figure out
*which sensors* — and therefore which manufacturing steps — are responsible.
That investigation is called **yield debugging** ("yield" = fraction of wafers
that pass), and it is one of the highest-leverage problems in fab operations:
a 1% yield improvement on a single line can be worth millions of dollars per
year.

This project automates the first hour of that investigation:

1. **Triage** — predict pass/fail for every wafer with calibrated probabilities.
2. **Explain** — for each wafer, identify the sensors that drove the model's
   decision (via SHAP) *and* show how far those readings sit from a healthy
   baseline (Z-score).
3. **Pattern-find** — cluster historical failures by their SHAP signatures to
   surface distinct **failure modes** instead of treating every defect alike.
4. **Monitor** — alert when a recent batch of wafers no longer looks like the
   data the model was trained on (PSI + KS drift tests).

All four are exposed through a five-page **Streamlit dashboard** (for engineers)
and a **FastAPI** REST service (for integration with MES / production systems).

---

## Dashboard

Run locally:

```bash
make app   # or: streamlit run app/streamlit_app.py
```

Or open the live demo on Streamlit Community Cloud (link in the repo's About
sidebar on GitHub). The dashboard is **zero-setup**: a 6 MB snapshot of the
trained model and processed data is committed to `dashboard_assets/`, so it
works immediately after cloning without running any notebooks.

![Dashboard Screenshot](docs/dashboard_screenshot.png?v=3)

### Five pages, five distinct use cases

| Page | Question it answers | What you see |
|---|---|---|
| **Wafer Diagnostic** | *Why does the model think this wafer fails?* | Pass/fail prediction, SHAP bar chart of top driver sensors, Z-score deviation chart, side-by-side overlay of the wafer vs. healthy normal range, plain-English summary, full feature table. |
| **Compare Wafers** | *Why did this wafer fail when that one passed?* | Side-by-side SHAP signatures and ranked Z-score differences for any two wafers. |
| **Aggregate Metrics** | *How good is the model overall — and what's the trade-off?* | Live-tunable decision threshold that updates the confusion matrix, precision/recall/F1/ROC-AUC, ROC + PR curves, and probability-distribution histogram in real time. |
| **Failure Clustering** | *Are all failures the same root cause, or distinct modes?* | K-means on SHAP signatures of every failed wafer, projected to 2-D via PCA. For each cluster, the top driver sensors are listed so you can name the failure mode. |
| **Drift Monitor** | *Has the production line changed since training?* | Per-sensor PSI + KS tests for a recent batch (slice of dataset, or upload your own CSV) vs. the training distribution, with severity-coded summary cards and a top-25 most-drifted-sensors chart. |

The copy throughout the dashboard is written for **non-technical readers** —
each section opens with a "When to use this view" caption, and any technical
term (SHAP, Z-score, PSI, precision/recall) is translated into plain English
at the point of use. A "How to read this page" expander on the metrics page
provides a one-screen glossary.

---

## Architecture

```
                      ┌──────────────────────┐
                      │  SECOM raw sensors   │
                      │  (1567 × 590, UCI)   │
                      └──────────┬───────────┘
                                 │ scripts/fetch_data.py  (sha256-verified)
                                 ▼
              src/preprocess.py  ─►  src/features.py
              (impute, drop                (correlation prune,
               constant, scale)             top-50 by mutual info)
                                 │
                                 ▼
                       src/train.py      ──── 5-fold stratified CV
                       (Random Forest          + F1-optimal threshold tuning
                       w/ class_weight)        + drift_reference.json
                                 │
                                 ▼
                  models/  +  dashboard_assets/  (committed snapshot)
                                 │
                ┌────────────────┼─────────────────┐
                ▼                ▼                 ▼
          app/streamlit_app   api/main.py      tests/  + CI
          (5-page UI,         (FastAPI:        (pytest, ruff,
           SHAP/PSI/KS)        /predict,        Docker smoke
                               /predict/batch,   test on every push)
                               /metadata,
                               /drift,
                               /health)
                                 │
                                 ▼
                       Dockerfile + docker-compose.yml
                       (3.11-slim base, healthcheck,
                        boots in ~5s with bundled model)
```

Key design decisions:

- **`src/artifacts.py` centralizes pipeline loading** — a single
  `load_pipeline_artifacts()` is consumed by the dashboard, the API, and the
  tests, so the four artifacts (variance selector, scaler, MI-selected feature
  list, RF model) and the tuned threshold can never drift between callers.
- **`dashboard_assets/` is intentionally tracked.** Notebooks and `train.py`
  write into gitignored `data/` and `models/` for full reproducibility, but
  a 6 MB snapshot is committed so the dashboard and Docker image work on a
  fresh clone with zero setup.
- **The Docker image is self-contained.** It copies
  `dashboard_assets/models/` into `/app/models/` at build time, so
  `docker build .` works without any prior training step. CI smoke-tests
  the container against `/health` and `/metadata` on every push.
- **Drift reference is a 50 KB decile summary**, not the full training set —
  so the API can compute PSI + KS at inference time without bundling 5 MB
  of training data into the image.

---

## Results

### Model comparison

Three approaches were compared, all tracked with [MLflow](https://mlflow.org/):

| Model | Type | ROC-AUC | Fail Recall | Fail F1 |
|---|---|---|---|---|
| Isolation Forest | Anomaly detection | 0.57 | 19% | 0.20 |
| **Random Forest** | Supervised classification | **0.80** | **67%** | **0.35** |
| LSTM Autoencoder | Deep learning (reconstruction) | 0.58 | 10% | 0.11 |

Random Forest wins decisively on this dataset, which is unsurprising: the failure
signal is concentrated in a handful of sensors with non-linear thresholds —
exactly what tree ensembles are good at — and there isn't enough data
(~100 failures) for the deep model to learn a reliable representation.

### Cross-validated headline numbers

A single 80/20 split is high-variance with only ~21 fail samples in the test
set, so `src/train.py` reports **5-fold stratified CV** alongside the held-out
metrics. From `dashboard_assets/models/cv_metrics.json`:

- **ROC-AUC**: 0.747 ± 0.044
- **PR-AUC** : 0.215 ± 0.034 (vs. 0.066 baseline = the failure rate itself)

PR-AUC is the more informative number on this 6.6%-positive dataset — the
model lifts precision-recall area-under-curve by roughly **3.3×** over a
random classifier.

### Threshold tuning

The decision threshold is tuned on the validation set to maximize Fail-F1.
The F1-optimal threshold of **0.222** roughly **doubles fail recall** (33% → 67%)
versus the default 0.5, at a small precision cost — a sensible trade-off in
manufacturing where missing a defect is more expensive than a false alarm.
The tuned value lives in `models/threshold.json` and is loaded automatically
by both the API and the dashboard.

### Explainability

A small subset of sensors drives most predictions — `sensor_103` and `sensor_59`
have the highest average SHAP impact across the dataset. Their behavior shows
non-linear threshold effects: the reading is fine until it crosses a critical
value, then failure risk jumps sharply (visible on the "Top Sensor Drivers"
panel of any failed wafer in the dashboard).

For full intended use, training data details, evaluation, and limitations,
see [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn (RandomForest, mutual-info selection, calibration helpers), SHAP, scipy (KS test), MLflow (experiment tracking) |
| Data / pipelines | numpy, pandas, joblib (artifact persistence) |
| Dashboard | Streamlit, Plotly (dark-themed, interactive) |
| API | FastAPI + uvicorn, Pydantic v2 schemas, structured request logging, CORS, X-Request-ID |
| Deployment | Docker (`python:3.11-slim`), `docker-compose`, healthchecks, sha256-verified data fetcher |
| Quality | pytest (50+ tests across 6 modules), ruff (lint + format), pre-commit hooks |
| CI/CD | GitHub Actions: matrix on Python 3.11 & 3.12, lint, test, Docker build + smoke test |
| Reproducibility | Makefile, `.python-version`, `pyproject.toml`, model card, dataset hash pinning |

---

## Repo structure

```
.
├── app/
│   └── streamlit_app.py           # 5-page Streamlit dashboard
├── api/
│   └── main.py                    # FastAPI service (5 endpoints + CORS + logging)
├── src/
│   ├── preprocess.py              # impute, drop constant, scale
│   ├── features.py                # correlation prune + mutual-info top-50
│   ├── models.py                  # RF builder, CV runner, threshold tuner
│   ├── train.py                   # train + CV + threshold tuning + drift baseline
│   ├── diagnostics.py             # baseline analysis, deviation scoring, summaries
│   ├── drift.py                   # PSI + KS, decile-summary persistence
│   └── artifacts.py               # single source of truth for loading pipeline
├── notebooks/
│   ├── 01_eda.ipynb               # exploratory data analysis
│   ├── 02_modeling.ipynb          # baseline models
│   ├── 03_model_comparison.ipynb  # IF / RF / LSTM-AE + MLflow
│   ├── 04_explainability.ipynb    # SHAP deep-dive on the RF
│   └── 05_yield_debug_analysis.ipynb  # failure pattern clustering
├── tests/                         # pytest suite (api, preprocess, features,
│   ├── test_api.py                #   diagnostics, models, drift) — runs in CI
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_diagnostics.py
│   ├── test_models.py
│   ├── test_drift.py
│   └── conftest.py
├── scripts/
│   └── fetch_data.py              # sha256-verified UCI SECOM downloader
├── dashboard_assets/              # COMMITTED 6 MB snapshot for zero-setup demos
│   ├── data/                      #   X_clean.csv, y.csv
│   └── models/                    #   rf_model.joblib, scaler, var_selector,
│                                  #   threshold.json, cv_metrics.json,
│                                  #   drift_reference.json
├── .github/workflows/ci.yml       # lint + test (Py 3.11 & 3.12) + Docker smoke test
├── .pre-commit-config.yaml        # ruff, trailing-whitespace, EOF, large-file checks
├── pyproject.toml                 # ruff config, pytest config
├── Dockerfile                     # python:3.11-slim, healthcheck, ~100 MB
├── docker-compose.yml             # one-command API stack
├── Makefile                       # install/test/lint/app/api/docker/snapshot/clean
├── MODEL_CARD.md                  # intended use, training data, metrics, limitations
├── requirements.txt               # runtime: dashboard + API
├── requirements-dev.txt           # + notebooks, training, tests, lint
└── README.md
```

---

## Setup

### Quick start (dashboard only — no setup)

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Or with the bundled `Makefile`:

```bash
make install   # runtime deps only
make app       # streamlit dashboard
```

The dashboard is **immediately functional** because the trained model and
processed data are checked in under `dashboard_assets/` (~6 MB). No data
download or training step required.

### Full pipeline (EDA → training → dashboard)

To reproduce the full workflow from scratch:

```bash
make install-dev          # 1. install all deps (notebooks, training, tests)
make fetch-data           # 2. sha256-verified download of SECOM into data/raw/
jupyter notebook notebooks/01_eda.ipynb   # 3. produces data/processed/
make train                # 4. produces models/ + threshold.json + drift_reference.json
make snapshot             # 5. (optional) refresh dashboard_assets/ from models/
make app                  # 6. launch dashboard
make api                  # 7. (optional) launch FastAPI on :8000
```

### Docker

A self-contained API image is wired up via `docker-compose.yml`:

```bash
make docker-up    # build + start the API on http://localhost:8000
curl http://localhost:8000/health
make docker-down
```

The Dockerfile bundles `dashboard_assets/models/` into the image, so
`docker build .` works on a fresh clone — no training required. The image is
~100 MB and boots in under 5 seconds.

---

## API

A [FastAPI](https://fastapi.tiangolo.com/) service mirrors the dashboard's
prediction logic as a REST API for integration with MES, dashboards, or
downstream systems.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check + lightweight pipeline metadata. |
| `GET /metadata` | Full pipeline & model info — version, feature counts, decision threshold, selected feature names. |
| `POST /predict` | Single-wafer prediction with top-K SHAP explanation. |
| `POST /predict/batch` | High-throughput batch predictions (no SHAP, vectorized). |
| `POST /drift` | PSI report for a recent batch vs. the training baseline. |

Example:

```http
POST /predict
{
    "features": [0.23, -1.1, 0.88, ...]   // 446 sensor readings
}
```

Returns:

```json
{
    "prediction": "FAIL",
    "probability": 0.87,
    "decision_threshold": 0.222,
    "top_contributing_features": [
        {"feature": "sensor_59",  "shap_value": 0.142},
        {"feature": "sensor_103", "shap_value": 0.098}
    ]
}
```

Run locally:

```bash
uvicorn api.main:app --reload --port 8000
# Configurable env vars: MODEL_DIR, LOG_LEVEL, CORS_ORIGINS, MAX_BATCH_SIZE
```

The API also:

- Adds an **`X-Request-ID`** header to every response.
- Emits **structured key-value logs** —
  `event=request request_id=... method=... path=... status=... latency_ms=...` —
  for easy aggregation in Datadog / CloudWatch / Loki.
- Honours **CORS** via `CORS_ORIGINS` (comma-separated, default `*` for dev).
- Exposes a **Pydantic v2** schema for every request/response (auto-docs at `/docs`).

---

## Drift monitoring

Manufacturing processes drift over time — sensor calibrations change,
raw-material lots vary, equipment ages. The dashboard's **Drift Monitor** page
and the API's `/drift` endpoint compute, per sensor:

- **PSI (Population Stability Index)** — bucketed KL-style score against the
  training distribution. Industry-standard thresholds: `< 0.10` stable,
  `0.10–0.25` moderate, `> 0.25` significant (consider retraining).
- **Two-sample Kolmogorov–Smirnov test** — non-parametric p-value for
  *"these two samples come from the same distribution"*.

The training run persists `models/drift_reference.json` (decile summary, ~50 KB)
so the API can compute PSI + KS at inference time without keeping the full
training set around.

---

## Engineering quality

- **Tests** — pytest suite across 6 modules (api, preprocess, features,
  diagnostics, models, drift); the `/predict`, `/predict/batch`, `/drift`,
  `/health`, and `/metadata` endpoints are all covered, including malformed
  input handling.
- **CI** — every push runs lint (`ruff`), the full pytest suite on Python
  **3.11 and 3.12** in parallel (`fail-fast: false`), then a **Docker job**
  that builds the image and smoke-tests `/health` and `/metadata` against
  the running container. Container logs are dumped on smoke-test failure
  via an `EXIT` trap so regressions surface the actual root cause.
- **Lint & format** — `ruff` configured in `pyproject.toml`; identical rules
  enforced locally via `.pre-commit-config.yaml` (ruff, trailing whitespace,
  EOF newline, YAML check, large-file guard, merge-conflict guard).
- **Reproducibility** — `Makefile` for one-command setup/test/run/build,
  sha256-verified data fetcher, supported Python pinned in **four canonical
  locations** (`.python-version`, `pyproject.toml`, `Dockerfile`, CI matrix),
  documented model card.
- **Observability** — structured single-line key-value logs (filterable on
  `event=`, `request_id=`, `path=`), per-request latency in milliseconds,
  Docker `HEALTHCHECK` pinging `/health` every 30 s.

---

## Model card

For full intended use, training data details, evaluation methodology,
limitations, ethical considerations, and reproduction steps, see
[`MODEL_CARD.md`](MODEL_CARD.md).

---

## Future work

- Integration with manufacturing execution systems (MES) for real-time wafer scoring.
- Build a library of named, labeled failure modes from past investigations to
  turn the unsupervised clustering into a supervised "what kind of failure is
  this?" classifier.
- Time-series-aware features (sensor trends per lot/shift), since SECOM only
  exposes a single snapshot per wafer.
- Model retraining pipeline triggered by drift alerts.
