from typing import Optional

import numpy as np
from sklearn.utils import check_random_state


def sample_ranking_fast_with_replacement(
    action_dist: np.ndarray,
    candidate_action_set_at_k: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=(action_dist.shape[0], action_dist.shape[2]))[
        :, np.newaxis
    ]
    cum_action_dist = action_dist.cumsum(axis=1)
    flg = cum_action_dist > uniform_rvs
    sampled_action_at_k = flg.argmax(axis=1)

    if candidate_action_set_at_k is None:
        return sampled_action_at_k

    position = np.arange(candidate_action_set_at_k.shape[0])
    sampled_slate = candidate_action_set_at_k[position, sampled_action_at_k].copy()

    return sampled_action_at_k, sampled_slate


def gumbel_softmax_sampling():
    pass
