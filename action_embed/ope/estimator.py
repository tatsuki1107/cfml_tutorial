from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Callable
from typing import Optional

import numpy as np
from obp.utils import check_array


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
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=weight, name="weight", expected_dim=1)
        
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
        delta: float = 0.05,
    ) -> tuple[np.float64]:
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=weight, name="weight", expected_dim=1)
        
        r_hat = self._estimate_round_rewards(reward=reward, weight=weight)
        cnf = lower_bound_func(r_hat, delta=delta, with_dev=True)
        return np.mean(r_hat), cnf


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    def _estimate_round_rewards(
        self, action_dist: np.ndarray, q_hat: np.ndarray
    ) -> np.ndarray:
        return np.average(q_hat, weights=action_dist, axis=1)

    def estimate_policy_value(
        self, action_dist: np.ndarray, q_hat: np.ndarray
    ) -> np.float64:
        check_array(array=action_dist, name="action_dist", expected_dim=2)
        check_array(array=q_hat, name="q_hat", expected_dim=2)
        
        return self._estimate_round_rewards(action_dist=action_dist, q_hat=q_hat).mean()


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
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

        return ips_values + dm_values

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        q_hat: np.ndarray,
        q_hat_factual: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=weight, name="weight", expected_dim=1)
        check_array(array=q_hat, name="q_hat", expected_dim=2)
        check_array(array=q_hat_factual, name="q_hat_factual", expected_dim=1)
        check_array(array=action_dist, name="action_dist", expected_dim=2)
        
        return self._estimate_round_rewards(
            reward=reward,
            weight=weight,
            q_hat=q_hat,
            q_hat_factual=q_hat_factual,
            action_dist=action_dist,
        ).mean()


@dataclass
class OFFCEM(BaseOffPolicyEstimator):
    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        f_hat: np.ndarray,
        f_hat_factual: np.ndarray,
        action_dist: np.ndarray,
        cluster_dist: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ips_values = weight * (reward - f_hat_factual)
        
        if cluster_dist is None:
            dm_values = np.average(f_hat, weights=action_dist, axis=1)
        
        else:
            action_dist_given_cluster = action_dist
            f_hat_x_c = (action_dist_given_cluster * f_hat[:, :, None]).sum(axis=1)
            dm_values = np.average(f_hat_x_c, weights=cluster_dist, axis=1)

        return ips_values + dm_values

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        weight: np.ndarray,
        f_hat: np.ndarray,
        f_hat_factual: np.ndarray,
        action_dist: Optional[np.ndarray],
        cluster_dist: Optional[np.ndarray] = None,
    ) -> float:
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=weight, name="weight", expected_dim=1)
        check_array(array=f_hat, name="f_hat", expected_dim=2)
        check_array(array=f_hat_factual, name="f_hat_factual", expected_dim=1)
        if cluster_dist is not None:
            check_array(array=cluster_dist, name="cluster_dist", expected_dim=2)
        
        return self._estimate_round_rewards(
            reward=reward,
            weight=weight,
            f_hat=f_hat,
            f_hat_factual=f_hat_factual,
            action_dist=action_dist,
            cluster_dist=cluster_dist,
        ).mean()
