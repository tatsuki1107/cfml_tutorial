from multiprocessing import Pool, cpu_count

import numpy as np
from obp.utils import softmax


def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.3,
    is_replacement: bool = True,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    if is_replacement:
        base_pol = np.zeros_like(expected_reward)
        a = (
            np.argmax(expected_reward, axis=1)
            if is_optimal
            else np.argmin(expected_reward, axis=1)
        )
        base_pol[
            np.arange(expected_reward.shape[0])[:, None],
            a,
            np.arange(expected_reward.shape[2])[None, :],
        ] = 1
        pol = (1.0 - eps) * base_pol
        pol += eps / expected_reward.shape[1]

    else:
        raise NotImplementedError

    return pol


def plackett_luce(policy_logit: np.ndarray, all_rankings: np.ndarray) -> np.ndarray:
    with Pool(cpu_count() - 1) as p:
        job_args = [(policy_logit_, all_rankings) for policy_logit_ in policy_logit]
        action_dist = list(p.imap(_compute_ranking_policy_per_context, job_args))

        return np.array(action_dist)


def _compute_ranking_policy_per_context(job_args: tuple[np.ndarray, ...]):
    policy_logit, all_rankings = job_args
    n_rankings, len_list = all_rankings.shape
    n_unique_action = len(policy_logit)

    unique_action_set_2d = np.tile(np.arange(n_unique_action), reps=(n_rankings, 1))

    ranking_idx = np.arange(n_rankings)
    action_dist = np.ones(n_rankings)
    for pos_ in range(len_list):
        mask = unique_action_set_2d == all_rankings[:, pos_][:, np.newaxis]
        action_index = np.where(mask)[1]
        action_dist *= softmax(policy_logit[unique_action_set_2d])[
            ranking_idx, action_index
        ]

        if pos_ + 1 != len_list:
            mask = np.ones((n_rankings, n_unique_action - pos_))
            mask[ranking_idx, action_index] = 0
            unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                (-1, n_unique_action - pos_ - 1)
            )

        return action_dist
