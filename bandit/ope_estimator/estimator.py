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
    def estimate(self, *args, **kwargs):
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
        for context, action, reward in zip(contexts, actions, rewards):
            tiled_contexts = np.tile(context, (self.n_action, 1))
            selected_action = self.policy.select_action(contexts=tiled_contexts)

            if selected_action == action:
                pi_new_rewards.append(reward)

                batch_data = [[context, action, reward, None]]
                self.policy.update_parameter(batch_data=batch_data)

        return np.mean(pi_new_rewards)


@dataclass
class DirectMethod(BaseEstimator):

    def estimate(
        self,
        contexts: np.ndarray,
        rewards_hats: np.ndarray,
    ) -> np.float64:

        pi_new_rewards = []
        for context, rewards_hat in zip(contexts, rewards_hats):
            tiled_contexts = np.tile(context, (self.n_action, 1))
            selected_action = self.policy.select_action(contexts=tiled_contexts)

            reward_hat = rewards_hat[selected_action]
            pi_new_rewards.append(reward_hat)

            batch_data = [[context, selected_action, reward_hat, None]]
            self.policy.update_parameter(batch_data=batch_data)

        return np.mean(pi_new_rewards)


@dataclass
class InversePropensityScore(BaseEstimator):
    """Estimate the reward using Inverse Propensity Score (IPS)."""

    def estimate(
        self,
        contexts: np.ndarray,
        pi_b_hats: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> np.float64:
        """Estimate the reward using Inverse Propensity Score (IPS).

        Args:
            contexts (np.ndarray): The context.
            pi_b (np.ndarray): probability of selecting the action.
            actions (np.ndarray): The action.
            rewards (np.ndarray): The reward.

        Returns:
            np.float64: The estimated reward.
        """

        pi_new_rewards = []
        for context, pi_b_hat, action, reward in zip(
            contexts, pi_b_hats, actions, rewards
        ):
            tiled_contexts = np.tile(context, (self.n_action, 1))
            selected_action = self.policy.select_action(contexts=tiled_contexts)

            if selected_action == action:
                pi_new_rewards.append(reward / pi_b_hat)

                batch_data = [[context, action, reward, None]]
                self.policy.update_parameter(batch_data=batch_data)

        return np.mean(pi_new_rewards)


@dataclass
class DoublyRobust(BaseEstimator):

    def estimate(
        self,
        contexts: np.ndarray,
        pi_b_hats: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        rewards_hats: np.ndarray,
    ) -> np.float64:

        pi_new_rewards = []
        for context, pi_b_hat, action, reward, rewards_hat in zip(
            contexts, pi_b_hats, actions, rewards, rewards_hats
        ):
            tiled_contexts = np.tile(context, (self.n_action, 1))
            selected_action = self.policy.select_action(contexts=tiled_contexts)
            reward_hat = rewards_hat[selected_action]
            binary_indicator = int(selected_action == action)

            _reward = binary_indicator * ((reward - reward_hat) / pi_b_hat) + reward_hat
            pi_new_rewards.append(_reward)

            batch_data = [[context, selected_action, _reward, None]]
            self.policy.update_parameter(batch_data=batch_data)

        return np.mean(pi_new_rewards)
