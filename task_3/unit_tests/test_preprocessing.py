import pandas as pd

from src.preprocessing import align_features, missing_value_indicators


def test_missing_value_indicators_adds_missing_columns():
    df = pd.DataFrame(
        {
            "avg_seller_lat": [10.0, None],
            "avg_seller_lng": [20.0, 21.0],
            "distance_km": [1.0, None],
            "max_product_weight_grams": [100, 200],
            "max_product_length_cm": [30, None],
            "max_product_height_cm": [20, 30],
            "max_product_width_cm": [10, 15],
            "avg_product_name_length": [30.0, 35.0],
            "avg_product_description_length": [100.0, None],
            "avg_product_photos_qty": [2.0, 3.0],
            "num_payments": [1, None],
            "num_installments": [1, 2],
            "total_payment_value": [50.0, None],
            "pay_boleto": [0, 1],
            "pay_credit_card": [1, 0],
            "pay_debit_card": [0, 0],
            "pay_not_defined": [0, 0],
            "pay_voucher": [0, 0],
            "customer_lat": [10.0, None],
            "customer_lng": [20.0, 21.0],
        }
    )

    out = missing_value_indicators(df)

    assert "avg_seller_lat_missing" in out.columns
    assert "distance_km_missing" in out.columns
    assert "customer_lat_missing" in out.columns
    assert out["avg_seller_lat_missing"].tolist() == [0, 1]


def test_align_features_adds_missing_columns_and_keeps_feature_order():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    feature_names = ["a", "b", "c"]

    out = align_features(df, feature_names)

    assert list(out.columns) == ["a", "b", "c"]
    assert out["c"].tolist() == [0, 0]
