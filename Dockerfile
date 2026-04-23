FROM python:3.11-slim

WORKDIR /app

# Pip cache-friendly: install deps before copying app code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, source modules, and the committed dashboard
# snapshot which doubles as the API's default model_dir. This means
# `docker build .` works on a fresh clone without needing to retrain.
COPY src/ src/
COPY api/ api/
COPY dashboard_assets/models/ models/

ENV MODEL_DIR=/app/models \
    LOG_LEVEL=INFO \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
