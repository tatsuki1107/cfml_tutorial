from dataclasses import dataclass

import numpy as np


@dataclass
class InversePropensityScore:
    """Inverse Propensity Score Estimator Class."""

    estimator_name: str

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""

        # iw * r
        return weight * reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(
            reward=reward,
            weight=weight,
        ).mean()
