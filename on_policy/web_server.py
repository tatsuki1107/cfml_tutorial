from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
from time import time

import numpy as np


@dataclass
class BaseWebServer(ABC):
    def request(self, *args, **kwargs):
        pass

    @abstractmethod
    def response(self, *args, **kwargs):
        pass


@dataclass
class WebServer(BaseWebServer):
    n_arm: int

    def __post_init__(self) -> None:

        self.mu = np.random.uniform(0, 1, size=self.n_arm)
        self.max_mu = self.mu.max()

    def response(self, arm: int) -> Tuple[np.float64, np.float64]:
        """Play a slot machine.

        Args:
            arm (int): The arm to play.

        Returns:
            np.float64: The reward and regret.
        """
        regret = self.max_mu - self.mu[arm]
        reward = np.random.binomial(p=self.mu[arm], n=1)

        return reward, regret


@dataclass
class ContextualWebServer(BaseWebServer):
    n_action: int
    dim_action_context: int
    dim_user_context: int
    dim_context: int
    reward_type: str  # "binary" or "continuous"
    noise_ver: float

    def __post_init__(self) -> None:
        """Initialize the action contexts and the parameters."""
        # i.i.d
        self.theta = np.random.normal(0, 1, size=(self.n_action, self.dim_context))

        if self.reward_type not in {"binary", "continuous"}:
            raise ValueError("reward_type must be 'binary' or 'continuous'")

    def request(self, t: int) -> np.ndarray:
        """Get the contexts based on the time step.

        Args:
            t (int): The time step.

        Returns:
            np.ndarray: The contexts.
        """

        np.random.seed(t)
        user_context = np.random.normal(0, 1, size=(self.dim_user_context))
        return user_context

    def response(
        self, contexts: np.ndarray, selected_action: np.int64
    ) -> Tuple[np.float64, np.float64]:
        """Play a slot machine.

        Args:
            contexts (np.ndarray): The contexts.
            selected_action (np.int64): The selected action.

        Returns:
            np.float64: The reward and regret.
        """
        np.random.seed(int(time()))
        # calculate reward
        counterfactual_rewards = []
        for action in range(self.n_action):
            mu = np.dot(contexts[action], self.theta[action])
            if self.reward_type == "binary":
                reward = np.random.binomial(n=1, p=sigmoid(mu))
            elif self.reward_type == "continuous":
                reward = np.random.normal(mu, self.noise_ver)

            counterfactual_rewards.append(reward)

        counterfactual_rewards = np.array(counterfactual_rewards)

        # regret
        regret = counterfactual_rewards.max() - counterfactual_rewards[selected_action]

        return counterfactual_rewards[selected_action], regret


def sigmoid(x: np.ndarray) -> np.ndarray:
    """The sigmoid function.

    Args:
        x (np.ndarray): The input.

    Returns:
        np.ndarray: The output.
    """

    return np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))
