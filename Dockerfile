FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt \
    && pip install --no-cache-dir --no-deps xgboost==3.4.1
COPY src/wildfire_recovery_forecasting/__init__.py src/wildfire_recovery_forecasting/api.py ./src/wildfire_recovery_forecasting/
RUN useradd --uid 10001 --create-home appuser
USER 10001
ENV WILDFIRE_MODEL_PATH=/model/model.pkl
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "wildfire_recovery_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
