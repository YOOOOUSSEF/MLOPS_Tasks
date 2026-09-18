import numpy as np
import pandas as pd


def predict(model, features: pd.DataFrame) -> dict:
    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0, 1])
    else:
        probability = float("nan")

    return {
        "prediction": prediction,
        "probability": probability,
    }