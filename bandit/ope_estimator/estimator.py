from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from policy import LinThompsonSampling, LinUCB, LogisticThompsonSampling


@dataclass
class BaseEstimator(ABC):
    n_action: int
    policy: Union[LinThompsonSampling, LinUCB, LogisticThompsonSampling]

    @abstractmethod
    def estimate(
        contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.float64:
        pass


@dataclass
class ReplayMethod(BaseEstimator):
    """Estimate the reward using the replay method.

    Args:
        n_action (int): The number of actions.

    Return:
        np.float64: The estimated reward.

    """

    def estimate(
        self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.float64:
        """Estimate the reward using the replay method.

        Args:
            contexts (np.ndarray): The context.
            actions (np.ndarray): The action.
            rewards (np.ndarray): The reward.

        Returns:
            np.float64: The estimated reward.
        """

        pi_new_rewards = []
        for t, (context, action, reward) in enumerate(zip(contexts, actions, rewards)):
            tiled_contexts = np.tile(context, (self.n_action, 1))
            selected_action = self.policy.select_action(
                contexts=tiled_contexts, t=t + 1
            )

            if selected_action == action:
                pi_new_rewards.append(reward)

                batch_data = [[context, action, reward, None]]
                self.policy.update_parameter(batch_data=batch_data)

        return np.mean(pi_new_rewards)
