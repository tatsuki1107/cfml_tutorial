from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseSlateInversePropensityScore(metaclass=ABCMeta):
    """Base class for Slate OPE estimators."""

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

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError


@dataclass
class SlateStandardInversePropensityScore(BaseSlateInversePropensityScore):
    """Slate Standard Inverse Propensity Score (SIPS) Estimator."""

    estimator_name: str = "sips"

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


@dataclass
class SlateIndependentInversePropensityScore(BaseSlateInversePropensityScore):
    """Slate Independent Inverse Propensity Score (IIPS) Estimator."""

    estimator_name: str = "iips"

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        alpha: np.ndarray,
        marginal_behavior_policy_pscore_at_position: np.ndarray,
        marginal_evaluation_policy_pscore_at_position: np.ndarray,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(
            reward=reward,
            alpha=alpha,
            behavior_policy_pscore=marginal_behavior_policy_pscore_at_position,
            evaluation_policy_pscore=marginal_evaluation_policy_pscore_at_position,
        ).mean()
