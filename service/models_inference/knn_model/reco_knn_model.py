import typing as tp

import implicit
import numpy as np
import pandas as pd

from service.models_inference.knn_model.download_artifact_knn import (
    DownloadArtifact,
)
from service.utils import Columns


class RecommendUserKNN:

    def __init__(self):

        loader = DownloadArtifact()

        self.type_reco = loader.get_type_reco()

        if self.type_reco == "offline":
            self.artifact = loader.get_offline_artifact()

        else:
            self.blending = loader.get_type_online_reco()

            if not self.blending:
                self.artifact = loader.get_online_artifact()

            else:
                self.artifact = loader.get_online_blending_artifact()

    @staticmethod
    def _get_sim_user(
        user: int,
        k_recs: int,
        model: implicit,
        users_mapping: tp.Dict[int, int],
        users_inv_mapping: tp.Dict[int, int],
        bmp: bool,
    ) -> pd.DataFrame:
        """
            The function find similar users.
        """
        user_id = users_mapping[user]
        recs, scores = model.similar_items(user_id, N=k_recs)

        if bmp:
            recs = list(
                filter(lambda x: x != users_inv_mapping[user_id], recs))

        else:
            start_idx = abs(
                len(list(filter(lambda x: x < 1, scores))) - len(scores)
            )
            recs = recs[start_idx:]

        return [users_inv_mapping[int(user)] for user in recs]

    def _get_offline_reco(
        self,
        user_id: int,
        k_recs: int,
    ) -> tp.List[int]:
        """
            The function creates offline recommendation.
        """
        offline_reco = self.artifact["offline_reco"]

        recs = list()
        if user_id in offline_reco[Columns.User]:
            recs = offline_reco[
                offline_reco[Columns.User] == user_id
                ][Columns.Item].tolist()

            if len(recs) > k_recs:
                recs = recs[:k_recs]

        return recs

    def _get_online_reco(
        self,
        user_id: int,
        k_recs: int,
        model: implicit = None,
        bmp=None,
        blending: bool = False,
    ) -> tp.List[int]:
        """
        The function creates online recommendation.
        """
        recs = list()
        if user_id in self.artifact["users_mapping"]:
            if bmp is None:
                bmp = self.artifact["bmp"]
            if model is None:
                model = self.artifact["model"]

            sim_user_id = self._get_sim_user(
                user_id,
                k_recs,
                model=model,
                users_mapping=self.artifact["users_mapping"],
                users_inv_mapping=self.artifact["users_inv_mapping"],
                bmp=bmp
            )

            recs = np.array(
                [item for user in sim_user_id for item in
                 self.artifact["watched"][user]]
            )
            recs = recs[
                np.sort(np.unique(recs, return_index=True)[1])].tolist()

            if (len(recs) > k_recs) and (not blending):
                recs = recs[:k_recs]

        return recs

    def _get_online_blending_reco(
        self,
        user_id: int,
        k_recs: int,
    ) -> tp.List[int]:
        """
            The function creates online recommendation
            with blending by tfidf algorithm.
        """
        recs = list()
        if user_id in self.artifact["users_mapping"]:

            recs_tfidf = self._get_online_reco(
                user_id=user_id,
                k_recs=k_recs,
                model=self.artifact["model_tfidf"],
                bmp=False,
                blending=True,
            )

            recs_bmp = self._get_online_reco(
                user_id=user_id,
                k_recs=k_recs,
                model=self.artifact["model_tfidf"],
                bmp=True,
                blending=True,
            )

            recs = np.unique(
                np.concatenate(
                    (recs_tfidf, recs_bmp)
                )
            )

            item_idf = self.artifact["item_idf"]
            mask = np.in1d(item_idf, recs)
            recs = item_idf[mask].tolist()

            if len(recs) > k_recs:
                recs = recs[:k_recs]

        return recs

    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:
        if self.type_reco == "offline":
            return self._get_offline_reco(user_id, k_recs)

        else:
            if not self.blending:
                return self._get_online_reco(user_id, k_recs)

            else:
                return self._get_online_blending_reco(user_id, k_recs)
