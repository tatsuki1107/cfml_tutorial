from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
from time import time

import numpy as np

# web server


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
    dim_user_context: int
    dim_context: int
    reward_type: str  # "binary" or "continuous"
    noise_ver: float

    def __post_init__(self) -> None:
        """Initialize the action contexts and the parameters."""

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
    ) -> Tuple[np.float64, np.float64, np.ndarray]:
        """Play a slot machine.

        Args:
            contexts (np.ndarray): The contexts.
            selected_action (np.int64): The selected action.

        Returns:
            np.float64: The reward and regret.
        """
        np.random.seed(int(time()))
        # calculate reward
        counterfactual_expected_reward = []
        for action in range(self.n_action):
            mu = np.dot(contexts[action], self.theta[action])

            if self.reward_type == "binary":
                mu = sigmoid(mu)
            counterfactual_expected_reward.append(mu)

        counterfactual_expected_reward = np.array(counterfactual_expected_reward)
        expected_reward = counterfactual_expected_reward[selected_action]

        if self.reward_type == "binary":
            reward = np.random.binomial(n=1, p=expected_reward)
        elif self.reward_type == "continuous":
            reward = np.random.normal(expected_reward, self.noise_ver)

        # regret
        regret = counterfactual_expected_reward.max() - expected_reward

        return reward, regret


@dataclass
class MFWebServer(BaseWebServer):
    n_user: int
    n_action: int
    dim_context: int
    noise_ver: float
    reward_type: str

    def __post_init__(self) -> None:
        if self.reward_type not in {"binary", "continuous"}:
            raise ValueError("reward_type must be 'binary' or 'continuous'")

        self.user_embedding = np.random.multivariate_normal(
            mean=np.zeros(self.dim_context),
            cov=np.eye(self.dim_context),
            size=self.n_user,
        )
        self.action_embedding = np.random.multivariate_normal(
            mean=np.zeros(self.dim_context),
            cov=np.eye(self.dim_context),
            size=self.n_action,
        )

    def request(self, t: int) -> np.ndarray:
        """Get the contexts based on the time step.

        Args:
            t (int): The time step.

        Returns:
            np.ndarray: The contexts.
        """

        np.random.seed(t)
        return np.random.randint(0, self.n_user)

    def response(
        self, user: int, selected_action: np.int64
    ) -> Tuple[np.float64, np.float64]:
        """Play a slot machine.

        Args:
            user (int): The user.

        Returns:
            np.float64: The reward and regret.
        """
        np.random.seed(int(time()))
        # calculate reward
        counterfactual_rewards = []
        for action in range(self.n_action):
            mu = np.dot(self.user_embedding[user], self.action_embedding[action])
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


# backend server


def generate_action_context(n_action: int, dim_context: int) -> np.ndarray:
    """Generate the action context.

    Args:
        n_action (int): The number of actions.
        dim_context (int): The dimension of the context.

    Returns:
        np.ndarray: The action context.
    """
    return np.random.normal(0, 1, size=(n_action, dim_context))
