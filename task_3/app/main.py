from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from pathlib import Path
import yaml
from src.pipeline import InferencePipeline

app = FastAPI()


class order(BaseModel):
    order_id: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: int


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
