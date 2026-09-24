from pathlib import Path
import time
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

from src.data_access import (
    load_feature_list,
    load_pickle,
    load_model,
    create_order_features,
)
from src.features import add_date_features, add_distance
from src.preprocessing import align_features, apply_imputer
from src.prediction import predict
from src.validation import validate_input
from src.preprocessing import drop_unused_columns, missing_value_indicators
from src.logger import setup_logger

logger = setup_logger(__name__)


class InferencePipeline:
    def __init__(
        self,
        model_path: str,
        imputer_path: Path,
        feature_list_path: Path,
    ):
        self.model = load_model(model_path)
        self.imputer = load_pickle(imputer_path)
        self.feature_names = load_feature_list(feature_list_path)

    def prepare_features(self, order_ids: list[str]) -> pd.DataFrame:
        logger.info("Preparing features for prediction")
        data = create_order_features(order_ids)
        data = drop_unused_columns(data)
        data = add_date_features(data)
        data = add_distance(data)
        data = missing_value_indicators(data)

        data = apply_imputer(data, self.imputer)

        data = align_features(data, self.feature_names, self.model)

        validate_input(data)
        return data

    def predict(self, order_ids: list[str]) -> list[dict]:
        try:
            if not isinstance(order_ids, list):
                logger.error(
                    "Invalid input type: expected str, got %s", type(order_ids).__name__
                )
                raise ValueError(
                    "Input data must be a list[str] representing an order_ids."
                )

            logger.info("Prediction request Received")
            start = time.time()

            features = self.prepare_features(order_ids)
            prediction = predict(self.model, features)

            latency = time.time() - start

            load_dotenv()
            connection_string = os.getenv("DB_CONNECTION_STRING")
            engine = create_engine(connection_string)

            for index, doc in enumerate(prediction):
                query = text("""
                    INSERT INTO prediction_logs (order_id, prediction)
                    VALUES (:order_id, :prediction)
                    """)
                with engine.connect() as conn:
                    conn.execute(
                        query,
                        {
                            "order_id": order_ids[index],
                            "prediction": doc["prediction"],
                        },
                    )
                    conn.commit()

            logger.info(
                "Prediction request: input=%s output=%s latency=%.3fs model_version=1",
                features,
                prediction,
                latency,
            )

            return prediction
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            raise
