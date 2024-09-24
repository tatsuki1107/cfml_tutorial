from typing import Optional

import numpy as np

from utils.user_behavior import parse_behavior


def vanilla_weight(
    data: dict, action_dist: np.ndarray, user_idx: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    if user_idx is None:
        user_idx = rounds

    if data["is_replacement"]:
        position = np.arange(data["len_list"])[None, :]
        position_wise_weight = (
            action_dist[rounds[:, None], data["action_id_at_k"], position]
            / data["pscore"]
        )
        ranking_wise_weight = position_wise_weight.prod(1)
    else:
        ranking_wise_weight = action_dist[rounds, data["ranking_id"]] / data["pscore"]

    ranking_wise_weight = np.tile(ranking_wise_weight[:, None], reps=data["len_list"])

    return ranking_wise_weight[user_idx]


def independent_weight(
    data: dict, action_dist: np.ndarray, user_idx: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    if user_idx is None:
        user_idx = rounds

    if data["is_replacement"]:
        position = np.arange(data["len_list"])[None, :]
        position_wise_weight = (
            action_dist[rounds[:, None], data["action_id_at_k"], position]
            / data["pscore"]
        )
    else:
        raise NotImplementedError

    return position_wise_weight[user_idx]


def cascade_weight(
    data: dict, action_dist: np.ndarray, user_idx: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    if user_idx is None:
        user_idx = rounds

    if data["is_replacement"]:
        position = np.arange(data["len_list"])[None, :]
        position_wise_weight = (
            action_dist[rounds[:, None], data["action_id_at_k"], position]
            / data["pscore"]
        )
        cascade_iw = []
        for pos_ in range(data["len_list"]):
            cascade_iw.append(
                position_wise_weight[:, : pos_ + 1].prod(axis=1, keepdims=True)
            )

        cascade_iw = np.concatenate(cascade_iw, axis=1)
    else:
        raise NotImplementedError

    return cascade_iw[user_idx]


def inverse_cascade_weight(
    data: dict, action_dist: np.ndarray, user_idx: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    if user_idx is None:
        user_idx = rounds

    if data["is_replacement"]:
        position = np.arange(data["len_list"])[None, :]
        position_wise_weight = (
            action_dist[rounds[:, None], data["action_id_at_k"], position]
            / data["pscore"]
        )
        inverse_cascade_iw = []
        for pos_ in range(data["len_list"]):
            inverse_cascade_iw.append(
                position_wise_weight[:, pos_:].prod(axis=1, keepdims=True)
            )

        inverse_cascade_iw = np.concatenate(inverse_cascade_iw, axis=1)
    else:
        raise NotImplementedError

    return inverse_cascade_iw[user_idx]


def top_k_cascade_weight(
    data: dict,
    action_dist: np.ndarray,
    k: int,
    user_idx: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    if user_idx is None:
        user_idx = rounds

    if data["is_replacement"]:
        position = np.arange(data["len_list"])[None, :]
        position_wise_weight = (
            action_dist[rounds[:, None], data["action_id_at_k"], position]
            / data["pscore"]
        )
        top_k_cascade_iw = []
        for top_k_pos_ in range(k):
            top_k_cascade_iw.append(
                position_wise_weight[:, : top_k_pos_ + 1].prod(axis=1, keepdims=True)
            )
        for pos_ in range(k, data["len_list"]):
            top_k_cascade_iw.append(
                top_k_cascade_iw[-1] * position_wise_weight[:, [pos_]]
            )

        top_k_cascade_iw = np.concatenate(top_k_cascade_iw, axis=1)
    else:
        raise NotImplementedError

    return top_k_cascade_iw[user_idx]


def neighbor_k_weight(
    data: dict,
    action_dist: np.ndarray,
    k: int,
    user_idx: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    if user_idx is None:
        user_idx = rounds

    if data["is_replacement"]:
        position = np.arange(data["len_list"])[None, :]
        position_wise_weight = (
            action_dist[rounds[:, None], data["action_id_at_k"], position]
            / data["pscore"]
        )
        neighbor_k_iw = []
        for pos_ in range(data["len_list"]):
            top_k_neighbor, bottom_k_neighbor = (
                max(pos_ - k, 0),
                min(pos_ + k, data["len_list"]) + 1,
            )
            neighbor_k_iw.append(
                position_wise_weight[:, top_k_neighbor:bottom_k_neighbor].prod(
                    axis=1, keepdims=True
                )
            )

        neighbor_k_iw = np.concatenate(neighbor_k_iw, axis=1)
    else:
        raise NotImplementedError

    return neighbor_k_iw[user_idx]


CANDIDATE_WEIGHT_FUNC_DICT = {
    "standard": vanilla_weight,
    "independent": independent_weight,
    "cascade": cascade_weight,
    "inverse_cascade": inverse_cascade_weight,
    "top_k_cascade": top_k_cascade_weight,
    "neighbor_k": neighbor_k_weight,
}


def adaptive_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
    user_behavior = (
        kwargs["estimated_user_behavior"]
        if "estimated_user_behavior" in kwargs
        else data["user_behavior"]
    )

    adaptive_weight = np.zeros_like(data["action"], dtype=float)
    for behavior_name in np.unique(user_behavior):
        behavior_mask = user_behavior == behavior_name
        parsed_behavior = parse_behavior(behavior_name)
        if parsed_behavior:
            behavior_name_, k_ = parsed_behavior
            weight_func = CANDIDATE_WEIGHT_FUNC_DICT[behavior_name_]
            weight_ = weight_func(
                data=data, action_dist=action_dist, user_idx=behavior_mask, k=k_
            )
        else:
            weight_func = CANDIDATE_WEIGHT_FUNC_DICT[behavior_name]
            weight_ = weight_func(
                data=data, action_dist=action_dist, user_idx=behavior_mask
            )

        adaptive_weight[behavior_mask] = weight_

    return adaptive_weight


def _compute_position_wise_weight(args):
    action_dist, pi_b, ranking, all_rankings, n_unique_action = args
    len_list = all_rankings.shape[1]

    position_wise_weight = [[] for _ in range(len_list)]
    for pos_ in range(len_list):
        for action in range(n_unique_action):
            action_indicator = all_rankings[:, pos_] == action
            w_x_a_k = action_dist[action_indicator].sum() / pi_b[action_indicator].sum()
            position_wise_weight[pos_].append(w_x_a_k)

    position_wise_weight = np.array(position_wise_weight)
    position_wise_weight_factual = position_wise_weight[:, ranking]
    return position_wise_weight_factual


def _compute_cascade_weight(args):
    pass
