import typing as tp

import pandas as pd
import yaml

from service.models_inference.knn_model.reco_knn_model import RecommendUserKNN
from service.models_inference.vector_model.reco_vector_model import \
    RecommendVectorModel
from service.models_inference.ranker_model.reco_ranker_model import RankerModel


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
        self.models = dict()

        """
        Download knn Models
        """
        self.models["knn_model"] = RecommendUserKNN()

        """
        Download ALS / LightFM Models
        """
        self.models["vector_model"] = RecommendVectorModel()

        """
        Download rankers
        """
        self.models["candidates"] = pd.read_pickle(
            self.type_model["two_stage"]["data_candidate"]
        )
        self.models["ranker_pointwise"] = RankerModel()

    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:

        if self.type_model["one_stage"]["run"]:
            return self.models.recommend(user_id, k_recs)

        elif self.type_model["two_stage"]["run"]:
            candidates = self.models["candidates"][
                self.models["candidates"]["user_id"] == user_id
            ].explode(
                column=["item_id", "lfm_score", "rank"]
            )[["item_id", "lfm_score", "rank"]]

            if len(candidates) == 0:
                return []

            return self.models[
                self.type_model["two_stage"]["model_ranker"]
            ].recommend(user_id, k_recs, candidates)

        else:
            return []
