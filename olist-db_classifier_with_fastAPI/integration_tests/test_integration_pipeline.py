from pathlib import Path
import pytest
import yaml

from src.config import get_project_root
from src.pipeline import InferencePipeline


def test_invalid_order_id():
    sample_order_id = ["test-order-001"]

    with pytest.raises(ValueError, match="NO ORDER"):
        project_root = get_project_root()
        config = yaml.safe_load(open(project_root / "config" / "params.yaml"))

        # os.chdir("tests")# you are now on d:/MlOps_tasks/olist-db_classifier_with_fastAPI/tests so paths can work.

        pipeline = InferencePipeline(
            model_path=config["paths"]["mlflow_model_uri"],
            imputer_path=Path(config["paths"]["imputer"]),
            feature_list_path=Path(config["paths"]["feature_list"]),
        )

        print(pipeline.predict(sample_order_id))
