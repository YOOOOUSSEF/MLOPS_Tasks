from src.pipeline import InferencePipeline
from src.config import get_project_root
import pandas as pd
import yaml
from pathlib import Path
import sys


def return_to_parent_path():
    project_root = Path.cwd()
    if not (project_root / "src").exists():
        project_root = project_root.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def test_notebook_parity():
    return_to_parent_path()
    PROJ = get_project_root()
    p = yaml.safe_load(open(PROJ / "config" / "params.yaml"))

    # os.chdir("tests")# you are now on d:/MlOps_tasks/task_3/tests so paths can work.

    pipeline = InferencePipeline(
        model_path=p["paths"]["mlflow_model_uri"],
        imputer_path=Path(p["paths"]["imputer"]),
        feature_list_path=Path(p["paths"]["feature_list"]),
    )

    # check a few order predictions and compare them with the notebook predictions
    orders_ids = [
        "9e6bc602a2466daa94736f31d1319c5d",
        "8fb4e4bd46f9802e9179726dd20019a5",
        "88a78af246c6c2db88e72b384796c98c",
        "d7c663de1470a2d3671791d3f093991c",
        "c0e57db4a4a6ef32aa28911d1c07df81",
    ]

    print("PROCESSING Batch of ORDERs")
    results = pipeline.predict(orders_ids)

    # sort probabilities and round them to 6 decimal places after point.
    def round_result(r):
        return {
            "prediction": r["prediction"],
            "probability": round(r["probability"], 6),
        }

    results_sorted = sorted(
        [round_result(r) for r in results], key=lambda x: x["probability"]
    )
    notebook_sorted = sorted(
        [
            round_result(r)
            for r in pd.read_csv(p["paths"]["notebook_predictions"]).to_dict(
                orient="records"
            )
        ],
        key=lambda x: x["probability"],
    )
    assert (
        notebook_sorted == results_sorted
    ), "Pipeline output does not match notebook predictions!"

    # check a single order prediction of 1
    order_id = ["e4309eeca1cfe3c6df5a187c38817a5c"]

    print("PROCESSING LAST ORDER")
    result = pipeline.predict(order_id)

    assert result[0]["prediction"] == 1, "can not predict is_late/1"
