from dataclasses import dataclass

import numpy as np
from obp.ope import BaseOffPolicyEstimator


@dataclass
class InversePropensityScore(BaseOffPolicyEstimator):
    """Inverse Propensity Score Estimator Of Ranking Policies Class."""

    estimator_name: str

    def __post_init__(self) -> None:
        if not self.estimator_name in {"SIPS", "IIPS", "RIPS", "AIPS (true)", "AIPS"}:
            raise ValueError

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        alpha: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        return (weight * alpha * reward).sum(1)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        alpha: np.ndarray,
        weight: np.ndarray,
    ) -> np.float64:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(
            reward=reward, alpha=alpha, weight=weight
        ).mean()

    def estimate_interval(self):
        pass


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust Estimator Of Ranking Policies"""

    estimator_name: str

    def __post_init__(self) -> None:
        if not self.estimator_name in {"Cascade-DR"}:
            raise ValueError

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        q_hat: np.ndarray,
        q_hat_factual: np.ndarray,
        action_dist: np.ndarray,
    ) -> np.ndarray:
        ips_values = weight * (reward - q_hat_factual)
        dm_values = np.average(q_hat, weights=action_dist, axis=1)

        return (ips_values + dm_values).sum(axis=1)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        q_hat: np.ndarray,
        q_hat_factual: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        return self._estimate_round_rewards(
            reward=reward,
            weight=weight,
            q_hat=q_hat,
            q_hat_factual=q_hat_factual,
            action_dist=action_dist,
        ).mean()
