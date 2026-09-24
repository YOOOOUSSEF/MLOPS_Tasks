from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import mlflow
from pathlib import Path
import yaml
from src.pipeline import InferencePipeline
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
from starlette.responses import Response
import time
from datetime import datetime, timedelta
from evidently import Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from src.logger import setup_logger

logger = setup_logger(__name__)


app = FastAPI()

LOOKBACK_DAYS = 7

# Metrics
metrics_registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Requests count",
    registry=metrics_registry,
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latency",
    registry=metrics_registry,
)

ERROR_COUNT = Counter("http_errors_total", "Errors", registry=metrics_registry)


class order(BaseModel):
    order_id: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: int


class DriftReportResponse(BaseModel):
    status: str


# ONCE APP started download model
PROJ = Path(__file__).parent.parent
p = yaml.safe_load(open(PROJ / "config" / "params.yaml"))
print(f"Project root: {PROJ}")

TRACKING_URI = f"sqlite:///{PROJ}/mlflow_tracker/mlflow.db"  # a single file, right here in the sandbox
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias("olist-classifier", "champion")

pipeline = InferencePipeline(
    model_path=p["paths"]["mlflow_model_uri"],
    imputer_path=Path(p["paths"]["imputer_fastapi"]),
    feature_list_path=Path(p["paths"]["feature_list_fastapi"]),
)


# Middleware (request count, latency, error rate)
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)

        # Count every request
        REQUEST_COUNT.inc()

        # Record latency
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)

        # Count 5xx errors
        if response.status_code >= 500:
            ERROR_COUNT.inc()

        return response

    except Exception:
        # Request raised an exception
        ERROR_COUNT.inc()

        # Still record request + latency
        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        raise


def get_current_predictions(engine) -> pd.DataFrame:
    """Pull recent predictions from prediction_logs."""
    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    query = text("""
        SELECT order_id, prediction, timestamp
        FROM prediction_logs
        WHERE timestamp >= :cutoff
        """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cutoff": cutoff})
        logger.info("Loaded %d reference predictions", len(df))
    return df


def get_reference_predictions() -> pd.DataFrame:
    """Load the reference (training-time) prediction distribution."""
    if not os.path.exists(p["paths"]["ref_data"]):
        raise FileNotFoundError(
            f"Reference data not found at {p['paths']['ref_data']}. "
            "Export your training/validation predictions once and save them there."
        )
    df = pd.read_csv(p["paths"]["ref_data"], usecols=["is_late"]).rename(
        columns={"is_late": "prediction"}
    )
    logger.info("Loaded %d reference predictions", len(df))
    return df


def run_drift_check(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Snapshot:
    """Run Evidently's data drift preset on the prediction column."""
    # Evidently expects matching column names in both dataframes.
    # We only care about the 'prediction' column for prediction drift.
    reference = reference_df[["prediction"]]
    current = current_df[["prediction"]]

    report = Report(metrics=[DataDriftPreset()])
    return report.run(reference_data=reference, current_data=current)


# Metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(metrics_registry),
        media_type="text/plain",
    )


@app.get(
    "/prediction_drift",
    response_model=DriftReportResponse,
    tags=["Monitoring"],
    summary="Generate the prediction drift report",
)
def make_report() -> DriftReportResponse:
    load_dotenv()
    connection_string = os.getenv("DB_CONNECTION_STRING")
    engine = create_engine(connection_string)

    current_df = get_current_predictions(engine)
    if current_df.empty:
        logger.warning(
            "No predictions found in the last %d days — skipping drift check.",
            LOOKBACK_DAYS,
        )
        return DriftReportResponse(status="skipped: no recent predictions")

    reference_df = get_reference_predictions()

    report = run_drift_check(reference_df, current_df)

    os.makedirs(os.path.dirname(p["paths"]["report"]), exist_ok=True)
    report.save_html(p["paths"]["report"])
    logger.info("Drift report saved to %s", p["paths"]["report"])
    return DriftReportResponse(status="saved")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model")
def model_info_version():
    return {
        "model_name": mv.name,
        "version": mv.version,
        "run_id": mv.run_id,
        "accuracy": client.get_run(mv.run_id).data.metrics["accuracy"] * 100,
        "input_schema": [
            {"name": col.name, "type": str(col.type)}
            for col in pipeline.model.metadata.get_input_schema().inputs
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(order: order) -> PredictionResponse:
    try:
        result = pipeline.predict([order.order_id])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    result[0]["model_version"] = mv.version
    return result[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(orders: list[order]) -> list[PredictionResponse]:
    order_ids = [o.order_id for o in orders]
    try:
        results = pipeline.predict(order_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    for r in results:
        r["model_version"] = mv.version
    return results
