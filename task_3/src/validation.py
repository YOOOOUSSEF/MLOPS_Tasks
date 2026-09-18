import pandas as pd
from src.logger import setup_logger

logger = setup_logger(__name__)

REQUIRED_COLUMNS = [
    "distance_km",
    "estimated_days",
    "customer_lat",
    "customer_lng",
    "avg_seller_lat",
    "avg_seller_lng",
]


def validate_input(data: pd.DataFrame) -> None:
    logger.info("Validating input data for required columns and non-empty DataFrame")
    try:
        missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if data.empty:
            raise ValueError("Input data cannot be empty")
    except Exception as e:
        logger.exception("ERROR IS:  %s", e)
        raise