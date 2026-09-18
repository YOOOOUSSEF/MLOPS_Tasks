from pathlib import Path

import yaml

from src.config import get_project_root
from src.pipeline import InferencePipeline


def test_pipeline_loads_and_predicts():
    project_root = get_project_root()
    config = yaml.safe_load(open(project_root / "config" / "params.yaml", encoding="utf-8"))

    pipeline = InferencePipeline(
        model_path=Path(config["paths"]["model"]),
        imputer_path=Path(config["paths"]["imputer"]),
        feature_list_path=Path(config["paths"]["feature_list"]),
    )

    sample_order = {
        "order_id": "test-order-001",
        "customer_id": "test-customer-001",
        "order_status": "delivered",
        "order_purchase": "2024-01-15",
        "order_approved": "2024-01-15",
        "order_delivered_carrier": "2024-01-20",
        "order_delivered_customer": "2024-01-25",
        "order_estimated_delivery": "2024-01-30",
    }

    result = pipeline.predict(sample_order)

    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"], int)
    assert 0 <= result["probability"] <= 1