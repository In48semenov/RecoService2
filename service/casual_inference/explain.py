import typing as tp

import dill
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix, csr_matrix

from service.utils import Columns
from service.utils.common_artifact import interactions


class ModelOutputExplain:
    explanation_cfg_path = "./service/configs/explanation.cfg.yml"

    def __init__(self, model_paths: tp.Dict[str, str], ):

        # Initialize explanation parameters
        with open(self.explanation_cfg_path) as file:
            explanation_params = yaml.safe_load(file)

        self.honestly = explanation_params["honestly"]
        self.min_score = explanation_params["min_score"]
        self.max_score = explanation_params["max_score"]
        self.text_template = explanation_params["text_template"]

        # Load Models
        self.models = dict()
        for model_name in model_paths:
            with open(model_paths[model_name], "rb") as file:
                self.models[model_name] = dill.load(file)

        # Load Interaction
        users_inv_mapping = dict(
            enumerate(interactions[Columns.User].unique())
        )
        self.users_mapping = {v: k for k, v in users_inv_mapping.items()}
        self.items_inv_mapping = dict(
            enumerate(interactions[Columns.Item].unique())
        )
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

        self.interactions_csr = self._create_interaction_csr_matrix(
            interactions,
        )

        # Load Items feature
        self.items = pd.read_csv(
            explanation_params["data"]["items"]["path"]
        )[explanation_params["data"]["items"]["columns"]]

    def _create_interaction_csr_matrix(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = Columns.User,
        item_col: str = Columns.Item,
        weight_col: str = None,
    ) -> csr_matrix:

        if weight_col:
            weights = interactions_df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(interactions_df), dtype=np.float32)

        interaction_matrix = coo_matrix((
            weights,
            (
                interactions_df[user_col].map(self.users_mapping.get),
                interactions_df[item_col].map(self.items_mapping.get)
            )
        ))

        return interaction_matrix.tocsr()

    def _als_explain(self, user_id: int, item_id: int) -> tp.Tuple[int, float]:
        total_score, top_contributions, _ = self.models["als"].explain(
            userid=0,
            user_items=self.interactions_csr[self.users_mapping[user_id], :],
            itemid=self.items_mapping[item_id],
            N=2,
        )

        top_contributions = top_contributions[1]

        if len(top_contributions) > 0:
            title_contributions = self.items[
                self.items[Columns.Item] == self.items_inv_mapping[
                    top_contributions[0]
                ]
                ]["title"].iloc[0]
            explanation = self.text_template[
                              "als"] + f"'{title_contributions}'"
        else:
            genres_contributions = self.items[
                self.items[Columns.Item] == self.items_inv_mapping[
                    top_contributions[0]
                ]
                ]["genres"].iloc[0]
            if genres_contributions != "no_genre":
                explanation = self.text_template[
                                  "dummy"] + f"'{genres_contributions}'"
            else:
                explanation = None

        return int(total_score * 100), explanation

    def _dummy_explain(self, item_id: int) -> tp.Tuple[int, str]:
        score = np.random.randint(self.min_score, self.max_score + 1)
        genres = self.items[
            self.items[Columns.Item] == item_id
            ]["genres"].iloc[0]

        if genres != "no_genre":
            explanation = self.text_template["dummy"] + f"'{genres}'"
        else:
            explanation = None

        return score, explanation

    def _post_processing(self, score, explanation):
        if not self.honestly and score < self.min_score:
            score = np.random.randint(self.min_score,
                                      self.max_score + 1)
        elif not self.honestly and score > self.max_score:
            score = self.max_score

        if explanation is None:
            explanation = self.text_template["popular"]

        return score, explanation

    def explain(
        self,
        model_name: str,
        user_id: int,
        item_id: int,
    ) -> tp.Tuple[int, str]:

        score, explanation = self._dummy_explain(item_id)
        if user_id in interactions[Columns.User]:
            if model_name == "als":
                score, explanation = self._als_explain(user_id, item_id)

        score, explanation = self._post_processing(score, explanation)

        return score, explanation
