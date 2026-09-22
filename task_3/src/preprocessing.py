import pandas as pd
from src.logger import setup_logger

logger = setup_logger(__name__)


def apply_imputer(data: pd.DataFrame, imputer) -> pd.DataFrame:
    logger.info("Applying imputer to missing values")
    try:
        missing_geo_cols = ["avg_seller_lat", "avg_seller_lng", "distance_km"]
        missing_dim_cols = [
            "max_product_weight_grams",
            "max_product_length_cm",
            "max_product_height_cm",
            "max_product_width_cm",
        ]
        missing_text_cols = [
            "avg_product_name_length",
            "avg_product_description_length",
            "avg_product_photos_qty",
        ]
        missing_payment_cols = [
            "num_payments",
            "num_installments",
            "total_payment_value",
            "pay_boleto",
            "pay_credit_card",
            "pay_debit_card",
            "pay_not_defined",
            "pay_voucher",
        ]
        missing_customer_geo_cols = ["customer_lat", "customer_lng"]

        all_missing_cols = (
            missing_geo_cols
            + missing_dim_cols
            + missing_text_cols
            + missing_payment_cols
            + missing_customer_geo_cols
        )
        data[all_missing_cols] = imputer.transform(data[all_missing_cols])

        return pd.DataFrame(data, columns=data.columns)
    except Exception as e:
        logger.exception("Failed to apply imputer to missing values: %s", e)
        raise


def align_features(
    data: pd.DataFrame, feature_names: list[str], model=None
) -> pd.DataFrame:
    logger.info("Aligning features to match model's expected input")
    try:
        result = data.copy()

        for feature_name in feature_names:
            if feature_name not in result.columns:
                result[feature_name] = 0

        result = result[feature_names]
        if model is not None:
            schema = model.metadata.get_input_schema()
            for col_spec in schema.inputs:
                name = col_spec.name
                dtype = col_spec.type.to_pandas()
                if name in result.columns:
                    if str(dtype).startswith("int") or str(dtype) == "int64":
                        result[name] = result[name].fillna(
                            0
                        )  # NaN one-hot/int columns → 0
                    result[name] = result[name].astype(dtype)
        print(f"result: {result}")
        return result
    except Exception:
        logger.exception("Failed to align features to match model's expected input")
        raise


# def add_Target_label(data: pd.DataFrame, target_label='is_late') -> pd.DataFrame:
# ml_row=data.copy()
# ml_row = ml_row[ml_row['order_delivered_customer'].notna()]
# date_cols = ['order_purchase','order_approved','order_delivered_carrier',
#            'order_delivered_customer','order_estimated_delivery',
#            'max_shipping_limit_date']
# ml_row[date_cols] = ml_row[date_cols].apply(pd.to_datetime)


####  ADD TARGET LABEL
# ml_row[target_label]=np.where(ml_row['order_delivered_customer']>ml_row
# ['order_estimated_delivery'],1,0)

# return ml_row


def drop_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping unused columns from the dataset")
    try:
        ml_row = data.copy()

        ml_row = ml_row[ml_row["order_delivered_customer"].notna()]
        ml_row = ml_row.drop(
            columns=[
                "order_id",
                "customer_id",
                "customer_unique_id",
                "seller_zipcode",
                "customer_zipcode",
                "customer_city",
                "seller_city",
                "max_shipping_limit_date",
                "order_delivered_carrier",
                "order_delivered_customer",
                "order_approved",
                "order_status",
            ]
        )
        ml_row = pd.get_dummies(
            ml_row, columns=["customer_state"], prefix="cust_state", dtype=int
        )

        return ml_row

    except Exception:
        logger.exception("Failed to drop unused columns from the dataset")
        raise


def missing_value_indicators(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating missing value indicators for specific columns")
    try:
        ml_row = data.copy()
        missing_geo_cols = ["avg_seller_lat", "avg_seller_lng", "distance_km"]
        missing_dim_cols = [
            "max_product_weight_grams",
            "max_product_length_cm",
            "max_product_height_cm",
            "max_product_width_cm",
        ]
        missing_text_cols = [
            "avg_product_name_length",
            "avg_product_description_length",
            "avg_product_photos_qty",
        ]
        missing_payment_cols = [
            "num_payments",
            "num_installments",
            "total_payment_value",
            "pay_boleto",
            "pay_credit_card",
            "pay_debit_card",
            "pay_not_defined",
            "pay_voucher",
        ]
        missing_customer_geo_cols = ["customer_lat", "customer_lng"]

        all_missing_cols = (
            missing_geo_cols
            + missing_dim_cols
            + missing_text_cols
            + missing_payment_cols
            + missing_customer_geo_cols
        )

        for col in all_missing_cols:
            ml_row[f"{col}_missing"] = ml_row[col].isna().astype(int)
            if ml_row[f"{col}_missing"].sum() > 0:
                logger.warning(
                    "Missing values found in column '%s'. Created indicator column '%s_missing'.",
                    col,
                    col,
                )

        return ml_row
    except Exception:
        logger.exception(
            "Failed to create missing value indicators for specific columns"
        )
        raise
