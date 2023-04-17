import dill
import typing as tp

# from rectools import Columns
import numpy as np
import pandas as pd
import yaml


class RankerModel:
    path_config_run = "./service/configs/inference-vector-model.cfg.yml"

    def __init__(self):
        """
        Download model artifacts.
        """
        with open(self.path_config_run) as models_config:
            params = yaml.safe_load(models_config)

        with open(params["model_path"], "rb") as file:
            self.model = dill.load(file)

        self.users_features = pd.read_csv(
            params["data_user"]["path_users_features"])
        self.items_features = pd.read_csv(
            params["data_user"]["path_items_features"])

        self.column_features = params["data_user"]["columns"]
        self.users_columns = self.users_features.columns

    def recommend(self,
                  user_id: int,
                  k_recs: int,
                  candidates: pd.DataFrame):

        users_features = pd.DataFrame(
            np.repeat(
                self.users_features[
                    self.users_features["user_id"] == user_id
                    ].values, len(candidates), axis=0
            )
        )
        users_features.columns = self.users_columns

        candidates_features = pd.merge(self.items_features,
                                       candidates,
                                       on="item_id",
                                       how="inner")

        data = pd.concat(
            [
                users_features,
                candidates_features,
            ],
            axis=1
        )[self.column_features]

        ranker_score = self.model.predict_proba(data)[:, 1]
        ranker_score = np.argsort(ranker_score)[::-1]

        recs = list(
            candidates_features["item_id"].values[ranker_score]
        )

        if len(recs) > k_recs:
            recs = recs[:k_recs]

        return recs
