import typing as tp

import pandas as pd
import yaml

from service.models_storage.knn_model.reco_knn_model import RecommendUserKNN
from service.models_storage.vector_model.reco_vector_model import \
    RecommendVectorModel
from service.models_storage.ranker_model.reco_ranker_model import RankerModel


class MainPipeline:
    """
    Class for recommend all pipeline recsys
    """

    path_pipeline = "./service/configs/pipeline.cfg.yml"

    def __init__(self):
        """
        Download type pipeline
        """
        with open(self.path_pipeline) as models_config:
            pipeline_params = yaml.safe_load(models_config)

        self.type_model = pipeline_params["type_model"]
        """
        Download knn Models
        """
        self.models = dict()
        self.models["knn_model"] = RecommendUserKNN()

        """
        Download ALS / LightFM Models
        """
        self.model["vector_model"] = RecommendVectorModel()

        """
        Download rankers
        """
        self.model["candidates"] = pd.read_csv(
            self.type_model["two_stage"]["data_candidate"]
        )
        self.model["ranker_pointwise"] = RankerModel()

    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:

        if self.type_model["one_stage"]["run"]:
            return self.model.recommend(user_id, k_recs)

        elif self.type_model["two_stage"]["run"]:
            candidates = self.model["candidates"][
                self.model["candidates"]["user_id"] == user_id
            ][["item_id", "lfm_score", "rank"]]

            return self.model[
                self.type_model["two_stage"]["model_ranker"]
            ].recommend(user_id, k_recs, candidates)

        else:
            return []
