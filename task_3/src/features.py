import numpy as np
import pandas as pd
from src.logger import setup_logger

logger = setup_logger(__name__)

def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding date features")
    try:
        result = data.copy()

        result["order_purchase"] = pd.to_datetime(result["order_purchase"])
        result["order_estimated_delivery"] = pd.to_datetime(
            result["order_estimated_delivery"]
        )
        result["purchase_month"] = result["order_purchase"].dt.month
        result["estimated_days"] = (
            result["order_estimated_delivery"] - result["order_purchase"]
        ).dt.days

        drop_cols = ['order_purchase', 'order_estimated_delivery']
        result = result.drop(columns=drop_cols)

        return result
    except Exception:
        logger.exception("Failed to add date features for")
        raise

def haversine(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2,
):
    logger.info("Calculating haversine distance")
    try:
        earth_radius_km = 6371

        latitude_1, longitude_1, latitude_2, longitude_2 = map(
            np.radians,
            [latitude_1, longitude_1, latitude_2, longitude_2],
        )

        delta_latitude = latitude_2 - latitude_1
        delta_longitude = longitude_2 - longitude_1

        value = (
            np.sin(delta_latitude / 2) ** 2
            + np.cos(latitude_1)
            * np.cos(latitude_2)
            * np.sin(delta_longitude / 2) ** 2
        )

        return 2 * earth_radius_km * np.arcsin(np.sqrt(value))
    except Exception:
        logger.exception("Failed to calculate haversine distance")
        raise


def add_distance(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating haversine distance")
    try:
        result = data.copy()

        result["distance_km"] = haversine(
            result["customer_lat"],
            result["customer_lng"],
            result["avg_seller_lat"],
            result["avg_seller_lng"],
        )

        return result
    except Exception:
        logger.exception("Failed to add distance feature")
        raise