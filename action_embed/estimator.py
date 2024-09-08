from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""
    
    estimator_name: str

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        raise NotImplementedError


@dataclass
class InversePropensityScore(BaseOffPolicyEstimator):
    """Inverse Propensity Score Estimator Class."""

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


@dataclass
class MarginalizedIPS(InversePropensityScore):

    def estimate_policy_value_with_dev(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        lower_bound_func: Callable,
        delta: float = 0.05
    ) -> tuple[np.float64]:
        r_hat = self._estimate_round_rewards(reward=reward, weight=weight)
        cnf = lower_bound_func(r_hat, delta=delta, with_dev=True)
        return np.mean(r_hat), cnf


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    def _estimate_round_rewards(self, action_dist: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
        return np.average(q_hat, weights=action_dist, axis=1)
    
    def estimate_policy_value(self, action_dist: np.ndarray, q_hat: np.ndarray) -> np.float64:
        return self._estimate_round_rewards(action_dist=action_dist, q_hat=q_hat).mean()


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        q_hat: np.ndarray,
        q_hat_factual: np.ndarray,
        action_dist: np.ndarray
    ) -> np.ndarray:
        ips_values = weight * (reward - q_hat_factual)
        dm_values = np.average(q_hat, weights=action_dist, axis=1)
        
        return ips_values + dm_values
    
    def estimate_policy_value(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        q_hat: np.ndarray,
        q_hat_factual: np.ndarray,
        action_dist: np.ndarray
    ) -> float:
        return self._estimate_round_rewards(
            reward=reward,
            weight=weight,
            q_hat=q_hat,
            q_hat_factual=q_hat_factual,
            action_dist=action_dist
        ).mean()
