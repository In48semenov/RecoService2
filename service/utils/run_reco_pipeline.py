import typing as tp

import yaml

from service.utils.user_knn.reco_userKNN import RecommendUserKNN
from service.utils.matrix_factorization.reco_mf import RecommendMF


class MainPipeline:
    """
    Class for recommend all pipeline recsys
    """

    path_pipeline = "./service/config/main-pipeline.cfg.yml"
    def __init__(self):

        """
        Download type pipeline
        """
        with open(self.path_pipeline) as models_config:
            pipeline = yaml.safe_load(models_config)

        self.type_model = pipeline["type_model"]

        if self.type_model["user_knn"]:
            self.model = RecommendUserKNN()

        elif self.type_model["matrix_factorization"]:
            self.model = RecommendMF()


    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:
        return self.model.recommend(user_id, k_recs)


pipeline = MainPipeline()



