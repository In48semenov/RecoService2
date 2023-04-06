import typing as tp

from service.utils.common_artifact import popular_items


def add_reco_popular(
    k_recs: int,
    curr_recs: tp.List[int],
) -> tp.List[int]:
    """
        The function adds popular to the recommendations,
        if this is not enough.
    """
    if len(curr_recs) < k_recs:
        curr_recs = set(curr_recs)
        for item_pop in popular_items:
            curr_recs.add(item_pop)
            if len(curr_recs) == k_recs:
                return list(curr_recs)
    else:
        return curr_recs
