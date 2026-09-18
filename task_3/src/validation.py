import pandas as pd


REQUIRED_COLUMNS = [
    "distance_km",
    "estimated_days",
    "customer_lat",
    "customer_lng",
    "avg_seller_lat",
    "avg_seller_lng",
]


def validate_input(data: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Input data cannot be empty")