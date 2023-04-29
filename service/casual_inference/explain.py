import dill
import typing as tp

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from service.utils.common_artifact import (
    interactions,
    users_inv_mapping, users_mapping,
    items_inv_mapping, items_mapping,
)


class ModelOutputExplain:

    items_path = "./service/data/kion_train/items.csv"

    text_template = {
        "dummy": "Вам могут быть интересны ",
        "als": "Рекомендую, тем кому нравится ",
    }

    def __init__(self, model_paths: tp.Dict[str, str],):
        self.models = dict()
        for model_name in model_paths:
            with open(model_paths[model_name], "rb") as file:
                self.models[model_name] = dill.load(file)

        self.interactions_csr = self._create_interaction_csr_matrix(
            interactions,
        )

        self.items = pd.read_csv(self.items_path)

    @staticmethod
    def _create_interaction_csr_matrix(
        interactions_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        weight_col: str = None,
    ) -> csr_matrix:

        if weight_col:
            weights = interactions_df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(interactions_df), dtype=np.float32)

        interaction_matrix = coo_matrix((
            weights,
            (
                interactions_df[user_col].map(users_mapping.get),
                interactions_df[item_col].map(items_mapping.get)
            )
        ))

        return interaction_matrix.tocsr()

    def _als_explain(self, user_id: int, item_id: int):
        total_score, top_contributions, _ = self.models["als"].explain(
            userid=users_mapping[user_id],
            user_items=self.interactions_csr[users_mapping[user_id], :],
            itemid=item_id,
            N=1,
        )

        title_contributions = self.items[
            self.items["title"] == items_inv_mapping[top_contributions[0][0]]
        ]

        text_output = self.text_template["als"] + title_contributions

        return int(total_score * 100), text_output

    def explain(
        self,
        model_name: str,
        user_id: int,
        item_id: int,
        honestly: bool = True,
    ) -> tp.Tuple[float, str]:

        if user_id in interactions["user_id"]:
            if model_name == "als":
                score, text = self._als_explain(user_id, item_id)

                if not honestly and score < 70:
                    score = np.random.randint(70, 98)

        else:
            score = np.random.randint(70, 98)
            text = self.text_template["text_template"] + self.items["genres"]

        return score, text
