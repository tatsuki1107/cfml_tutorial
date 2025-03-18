from dataclasses import dataclass
from abc import ABCMeta
from abc import abstractmethod

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
class TrajectoryWiseIS(BaseOffPolicyEstimator):
    def _estimate_round_rewards(
        self, reward: np.ndarray, weight: np.ndarray
    ) -> np.ndarray:
        return weight.prod(1) * reward.sum(1)

    def estimate_policy_value(self, reward: np.ndarray, weight: np.ndarray) -> float:
        check_array(array=reward, name="reward", expected_dim=2)
        check_array(array=weight, name="weight", expected_dim=2)

        return self._estimate_round_rewards(reward=reward, weight=weight).mean()


@dataclass
class StepWiseIS(BaseOffPolicyEstimator):
    def _estimate_round_rewards(
        self, reward: np.ndarray, weight: np.ndarray
    ) -> np.ndarray:
        iw = []
        for h in range(reward.shape[1]):
            is_1_to_h = weight[:, : h + 1].prod(axis=1, keepdims=True)
            iw.append(is_1_to_h)

        iw = np.concatenate(iw, axis=1)
        return (iw * reward).sum(axis=1)

    def estimate_policy_value(self, reward: np.ndarray, weight: np.ndarray) -> float:
        check_array(array=reward, name="reward", expected_dim=2)
        check_array(array=weight, name="weight", expected_dim=2)

        return self._estimate_round_rewards(reward=reward, weight=weight).mean()
