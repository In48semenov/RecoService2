import typing as tp

import dill
import numpy as np
import pandas as pd
import yaml


class DownloadArtifact:
    config_path = './service/config/inference-userKNN.cfg.yml'
    path_interactions_data = './data/kion_train/interactions.csv'
    path_item_idf = "./data/kion_train/items_idf.csv"

    def __init__(self):

        with open(self.config_path) as models_config:
            params = yaml.safe_load(models_config)

        self.run_params = params["run_params"]

    def _get_online_reco_artifact(self) -> tp.Dict:
        interactions = pd.read_csv(self.path_interactions_data)
        watched = interactions.groupby('user_id').agg(
            {'item_id': list}
        ).to_dict()["item_id"]
        users_inv_mapping = dict(enumerate(interactions['user_id'].unique()))
        users_mapping = {v: k for k, v in users_inv_mapping.items()}
        index_bmp_model = self.run_params["artifact"]["index_bmp_model"]

        return {
            "watched": watched,
            "users_inv_mapping": users_inv_mapping,
            "users_mapping": users_mapping,
            "index_bmp_model": index_bmp_model,
        }

    def _get_item_idf(self) -> np.array:
        return pd.read_csv(self.path_item_idf)["index"].values

    def _get_one_model(self, path_model: str = None):
        if path_model is None:
            path_model = self.run_params["artifact"]["model_path_1"]

        with open(path_model, "rb") as file:
            model = dill.load(file)

        return model

    def _get_several_model(self, k_model: int = 2) -> tp.List:
        models = list()
        for idx_model in range(1, k_model + 1):
            models.append(
                self._get_one_model(
                    path_model=self.run_params["artifact"][
                        f"model_path_{idx_model}"
                    ]
                )
            )

        return models

    def get_type_reco(self) -> str:
        return self.run_params["type_reco"]

    def get_type_online_reco(self) -> str:
        return self.run_params["artifact"]["blending"]

    def get_offline_artifact(self):
        return {
            "offline_reco": pd.read_csv(
                self.run_params["artifact"]["offline_reco_path"]
            )
        }

    def get_online_artifact(self):
        online_artifact = self._get_online_reco_artifact()
        bmp = True if online_artifact[
                          "index_bmp_model"
                      ] != -1 else False
        return {
            "model": self._get_one_model(),
            "watched": online_artifact["watched"],
            "users_mapping": online_artifact["users_mapping"],
            "users_inv_mapping": online_artifact["users_inv_mapping"],
            "bmp": bmp,
        }

    def get_online_blending_artifact(self):
        online_artifact = self._get_online_reco_artifact()

        models = self._get_several_model()
        bmp = True if online_artifact[
                          "index_bmp_model"
                      ] != -1 else False

        if bmp == 1:
            model_bmp = models[0]
            model_tfidf = models[1]
        else:
            model_bmp = models[1]
            model_tfidf = models[0]

        return {
            "model_tfidf": model_tfidf,
            "model_bmp": model_bmp,
            "item_idf": self._get_item_idf(),
            "watched": online_artifact["watched"],
            "users_mapping": online_artifact["users_mapping"],
            "users_inv_mapping": online_artifact["users_inv_mapping"],
        }
