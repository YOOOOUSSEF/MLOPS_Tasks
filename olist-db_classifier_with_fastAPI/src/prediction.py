import pandas as pd
from src.logger import setup_logger

logger = setup_logger(__name__)


def predict(model, features: pd.DataFrame) -> list[dict]:
    try:
        logger.info("Making predictions using the provided model")
        preds = model.predict(features)

        # unwrap pyfunc to get the raw sklearn model for predict_proba
        underlying_model = getattr(model, "_model_impl", None)
        if underlying_model is not None and hasattr(underlying_model, "predict_proba"):
            probas = underlying_model.predict_proba(features)[:, 1]
        else:
            probas = [float("nan")] * len(preds)  # if model has no probability

        results = [
            {"prediction": int(p), "probability": float(prob)}
            for p, prob in zip(preds, probas)
        ]
        return results
    except Exception:
        logger.exception("Failed to make predictions using the provided model")
        raise
