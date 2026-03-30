FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, source modules, and saved model artifacts
COPY src/ src/
COPY api/ api/
COPY models/ models/

ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
