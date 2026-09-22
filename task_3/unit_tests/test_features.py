import numpy as np
import pandas as pd

from src.features import add_date_features, add_distance, haversine


def test_add_date_features_extracts_month_and_days():
    df = pd.DataFrame(
        {
            "order_purchase": ["2024-01-15 10:00:00", "2024-02-20 12:30:00"],
            "order_estimated_delivery": ["2024-01-30 12:00:00", "2024-03-02 08:00:00"],
        }
    )

    out = add_date_features(df)

    assert "purchase_month" in out.columns
    assert "estimated_days" in out.columns
    assert out["purchase_month"].tolist() == [1, 2]
    assert out["estimated_days"].tolist() == [15, 10]
    assert "order_purchase" not in out.columns
    assert "order_estimated_delivery" not in out.columns


def test_haversine_returns_expected_distance_for_known_points():
    distance = haversine(0, 0, 0, 1)

    assert np.isclose(distance, 111.195, atol=1.0)


def test_add_distance_uses_haversine_and_keeps_row_shape():
    df = pd.DataFrame(
        {
            "customer_lat": [0.0],
            "customer_lng": [0.0],
            "avg_seller_lat": [0.0],
            "avg_seller_lng": [1.0],
        }
    )

    out = add_distance(df)

    assert "distance_km" in out.columns
    assert out.shape == (1, 5)
    assert np.isclose(out["distance_km"].iloc[0], 111.195, atol=1.0)
