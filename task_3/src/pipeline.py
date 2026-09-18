from pathlib import Path
import pandas as pd

from src.data_access import load_feature_list, load_pickle,create_order_features
from src.features import add_date_features, add_distance
from src.preprocessing import align_features,apply_imputer
from src.prediction import predict
from src.validation import validate_input
from src.preprocessing import apply_imputer
from src.preprocessing import drop_unused_columns, missing_value_indicators



class InferencePipeline:
    def __init__(
        self,
        model_path: Path,
        imputer_path: Path,
        feature_list_path: Path,
    ):
        self.model = load_pickle(model_path)
        self.imputer = load_pickle(imputer_path)
        self.feature_names = load_feature_list(feature_list_path)

    def prepare_features(self, order: dict) -> pd.DataFrame:
        data=create_order_features(order)
        data=drop_unused_columns(data)
        data = add_date_features(data)
        data = add_distance(data)
        data=missing_value_indicators(data)


        validate_input(data)


        data = apply_imputer(data,self.imputer)
        return align_features(data, self.feature_names)

    def predict(self, data: pd.DataFrame) -> dict:
        features = self.prepare_features(data)
        return predict(self.model, features)