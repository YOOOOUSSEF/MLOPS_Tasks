import numpy as np
import pandas as pd
from src.logger import setup_logger

logger = setup_logger(__name__)

def predict(model, features: pd.DataFrame) -> dict:
    logger.info("Making predictions using the provided model")
    try:
        prediction = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0, 1])
        else:
            probability = float("nan")

        return {
            "prediction": prediction,
            "probability": probability,
        }
    except Exception as e:
        logger.exception("Failed to make predictions using the provided model: %s", e)
        raise