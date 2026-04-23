# Convenience targets for SECOM Fault Detection.
# Most contributors will use: make install, make test, make app, make api.

PY := python
PIP := pip
UVICORN := uvicorn

DATA_DIR := data/raw
PROCESSED_DIR := data/processed
MODEL_DIR := models
DASHBOARD_ASSETS := dashboard_assets

.PHONY: help install install-dev test lint format fetch-data train app api docker docker-up docker-down clean snapshot

help:
	@echo "Targets:"
	@echo "  install       - install runtime dependencies"
	@echo "  install-dev   - install runtime + dev dependencies"
	@echo "  fetch-data    - download SECOM data from UCI (hash-verified)"
	@echo "  train         - run src/train.py to (re)build model artifacts"
	@echo "  test          - run pytest"
	@echo "  lint          - ruff check"
	@echo "  format        - ruff format + ruff check --fix"
	@echo "  app           - launch the Streamlit dashboard"
	@echo "  api           - launch the FastAPI service on :8000"
	@echo "  docker        - build the API Docker image"
	@echo "  docker-up     - docker compose up -d (API on :8000)"
	@echo "  docker-down   - docker compose down"
	@echo "  snapshot      - copy current models/ to dashboard_assets/"
	@echo "  clean         - remove caches and pyc files"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

fetch-data:
	$(PY) scripts/fetch_data.py --out-dir $(DATA_DIR)

train:
	$(PY) src/train.py

test:
	$(PY) -m pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

app:
	streamlit run app/streamlit_app.py

api:
	$(UVICORN) api.main:app --reload --port 8000

docker:
	docker build -t secom-fault-detection:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Refresh dashboard_assets from a freshly trained models/ directory.
# Run this after `make train` if you want the committed dashboard
# snapshot to track the latest model.
snapshot:
	@mkdir -p $(DASHBOARD_ASSETS)/models/preprocessing $(DASHBOARD_ASSETS)/models/feature_engineering
	@cp $(MODEL_DIR)/rf_model.joblib                              $(DASHBOARD_ASSETS)/models/
	@cp $(MODEL_DIR)/feature_names_input.json                     $(DASHBOARD_ASSETS)/models/
	@cp $(MODEL_DIR)/threshold.json                               $(DASHBOARD_ASSETS)/models/
	@cp $(MODEL_DIR)/cv_metrics.json                              $(DASHBOARD_ASSETS)/models/
	@cp $(MODEL_DIR)/drift_reference.json                         $(DASHBOARD_ASSETS)/models/
	@cp $(MODEL_DIR)/preprocessing/var_selector.joblib            $(DASHBOARD_ASSETS)/models/preprocessing/
	@cp $(MODEL_DIR)/preprocessing/scaler.joblib                  $(DASHBOARD_ASSETS)/models/preprocessing/
	@cp $(MODEL_DIR)/feature_engineering/corr_kept_cols.json      $(DASHBOARD_ASSETS)/models/feature_engineering/
	@cp $(MODEL_DIR)/feature_engineering/mi_selected_cols.json    $(DASHBOARD_ASSETS)/models/feature_engineering/
	@echo "Synced $(MODEL_DIR)/ -> $(DASHBOARD_ASSETS)/models/"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
