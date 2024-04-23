import numpy as np


def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.1,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    base_pol = np.zeros((expected_reward.shape[0], expected_reward.shape[1]))
    if is_optimal:
        a = np.argmax(expected_reward.mean(2), axis=1)
    else:
        a = np.argmin(expected_reward.mean(2), axis=1)
    base_pol[
        np.arange(expected_reward.shape[0]),
        a,
    ] = 1
    pol = (1.0 - eps) * base_pol
    pol += eps / expected_reward.shape[1]

    return pol
