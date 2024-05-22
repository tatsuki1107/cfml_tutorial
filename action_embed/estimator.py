from dataclasses import dataclass

import numpy as np


@dataclass
class InversePropensityScore:
    """Inverse Propensity Score Estimator Class."""

    estimator_name: str

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""

        # iw: \frac{\pi_e(・|x_i)}{\pi_b(・|x_i)}
        iw = evaluation_policy_pscore / behavior_policy_pscore

        # iw * r
        return iw * reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(
            reward=reward,
            behavior_policy_pscore=behavior_policy_pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        ).mean()
