from dataclasses import dataclass

import numpy as np


@dataclass
class SlateInversePropensityScore:
    """Slate Inverse Propensity Score Estimator Class."""

    estimator_name: str

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        alpha: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""

        # iw: \frac{\pi_e(・|x_i)}{\pi_b(・|x_i)}
        iw = evaluation_policy_pscore / behavior_policy_pscore

        # \sum_{k=1}^K iw \alpha_k \mathbb{r}_{i}(k)
        return (iw * alpha * reward).sum(1)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        alpha: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(
            reward=reward,
            alpha=alpha,
            behavior_policy_pscore=behavior_policy_pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        ).mean()
