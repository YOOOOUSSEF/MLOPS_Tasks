# Olist Late-Delivery API

This service loads the trained Olist late-delivery model and exposes predictions through FastAPI. Training is kept in the notebooks; the API performs inference and monitoring.

## Project structure

```text
olist-db_classifier_with_fastAPI/
├── app/main.py                 # FastAPI application and monitoring endpoints
├── src/                        # Inference, feature, data-access, and validation code
├── config/params.yaml          # Model, data, and report paths
├── data/                       # Raw, processed, and reference data
├── models/                     # Model and preprocessing artifacts
├── db/seed.sql                 # PostgreSQL schema and seed data
├── notebooks/                  # Reproducible analysis notebooks
├── unit_tests/                 # Unit tests
├── integration_tests/          # API and pipeline integration tests
├── Dockerfile                  # API image definition
├── docker-compose.yaml         # Local API and PostgreSQL services
└── reports/                    # Generated drift reports
```

## Run with Docker

From `olist-db_classifier_with_fastAPI/`, set `POSTGRES_PASSWORD` in your environment, then run:

```powershell
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`.

PostgreSQL is initialized from `db/seed.sql`. The database volume is persistent, so initialization scripts run only when the database is created for the first time.

## API

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `GET /health`
- Model information: `GET /model`
- Single prediction: `POST /predict` with `{"order_id": "..."}`
- Batch prediction: `POST /predict/batch`
- Metrics: `GET /metrics`
- Prediction drift report: `GET /prediction_drift`

Predictions are written to the PostgreSQL `prediction_logs` table. The drift endpoint compares recent predictions with `data/raw/ML_TABLE_LABELED.csv` and saves the report to `reports/drift_report.html`.

## Tests

Install development dependencies if needed:

```powershell
pip install -r requirements/development.txt
```

Run the unit tests:

```powershell
pytest unit_tests
```

The Docker runtime expects the model, imputer, feature list, reference data, and MLflow artifacts included in this repository.
