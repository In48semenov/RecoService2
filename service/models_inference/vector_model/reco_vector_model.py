import typing as tp

import nmslib
import numpy as np
import yaml

from service.utils.common_artifact import interactions


class RecommendVectorModel:
    """
    Class for recommendation method Matrix Factorization
    """
    path_config_run = "./service/configs/inference-vector-model.cfg.yml"

    def __init__(self):
        """
        Download model artifact
        """
        with open(self.path_config_run) as models_config:
            params = yaml.safe_load(models_config)

        self.user_embeddings = np.load(params["user_embeddings"])
        self.item_embeddings = np.load(params["item_embeddings"])

        approximate_search = params["approximate_search"]

        """
        Initialize index for approximate search
        """
        self.index = nmslib.init(
            method=approximate_search["method"],
            space=approximate_search["space_name"],
            data_type=nmslib.DataType.DENSE_VECTOR,
        )
        self.index.addDataPointBatch(self.item_embeddings)
        self.index.createIndex(approximate_search["index_time_params"])
        self.index.setQueryTimeParams(
            approximate_search["query_time_params"]
        )

        """
        Create item and user mapping
        """
        users_mapping = dict(enumerate(interactions["user_id"].unique()))
        self.users_inv_mapping = {v: k for k, v in users_mapping.items()}
        self.items_mapping = dict(
            enumerate(interactions["item_id"].unique())
        )

    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:
        """
        get reco
        """
        if user_id in self.users_inv_mapping:
            avatar_idx = self.users_inv_mapping[user_id]
            items_idx = self.index.knnQuery(
                self.user_embeddings[avatar_idx], k=k_recs
            )[0].tolist()
            return [self.items_mapping[idx] for idx in items_idx]
        else:
            return []
