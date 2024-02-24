from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class BaseBandit(ABC):
    """Base class for bandit algorithms.

    Args:
        mu (np.ndarray): True mean of each arm.
        T (int): Number of time steps.
    """

    mu: np.ndarray
    T: int

    def __post_init__(self) -> None:
        """Initialize the bandit algorithm."""

        self.N = np.zeros_like(self.mu, dtype=np.int64)
        self.rewards = np.zeros_like(self.mu)
        self.max_mu = self.mu.max()

    @abstractmethod
    def run(self) -> Tuple[List[np.float64], List[np.float64]]:
        """Run the bandit algorithm.
        Returns:
            Tuple[List[np.float64], List[np.float64]]:
            Cumulative reward and cumulative regret.
        """
        pass

    def _slot(self, arm: int) -> Tuple[np.float64, np.float64]:
        """Play a slot machine.

        Args:
            arm (int): The arm to play.

        Returns:
            np.float64: The reward and regret.
        """
        regret = self.max_mu - self.mu[arm]
        reward = np.random.binomial(p=self.mu[arm], n=1)
        self.N[arm] += 1
        self.rewards[arm] += reward
        return reward, regret


@dataclass
class EpsilonGreedy(BaseBandit):
    """Epsilon-greedy bandit algorithm.

    Args:
        epsilon (float): The exploration rate.
    """

    epsilon: float

    def run(self) -> Tuple[List[np.float64], List[np.float64]]:
        num_search_per_arm = int(self.epsilon * (self.T / len(self.mu)))
        num_utilization = self.T - int(num_search_per_arm * len(self.mu)) - 1

        cumulative_reward, cumulative_regret = [0], [0]

        for t in range(num_search_per_arm):
            for arm in range(len(self.mu)):
                reward, regret = self._slot(arm=arm)
                cumulative_reward.append(cumulative_reward[-1] + reward)
                cumulative_regret.append(cumulative_regret[-1] + regret)

        mu_hats = self.rewards / self.N
        mu_hat_max_arm = np.argmax(mu_hats)

        for t in range(num_utilization):
            reward, regret = self._slot(arm=mu_hat_max_arm)
            cumulative_reward.append(cumulative_reward[-1] + reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

        return cumulative_reward, cumulative_regret


@dataclass
class UCB(BaseBandit):
    """Upper confidence bound bandit algorithm."""

    def run(self) -> Tuple[List[np.float64], List[np.float64]]:
        cumulative_reward, cumulative_regret = [0], [0]

        # init
        for arm in range(len(self.mu)):
            reward, regret = self._slot(arm=arm)
            cumulative_reward.append(cumulative_reward[-1] + reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

        for t in range(len(self.mu) + 1, self.T):
            mu_hats = self.rewards / self.N
            ucb_scores = mu_hats + np.sqrt(np.log(t) / (2 * self.N))
            ucb_max_arm = np.argmax(ucb_scores)

            reward, regret = self._slot(arm=ucb_max_arm)
            cumulative_reward.append(cumulative_reward[-1] + reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

        return cumulative_reward, cumulative_regret


@dataclass
class Softmax(BaseBandit):
    """Softmax bandit algorithm.

    Args:
        tau (int): The temperature parameter.
    """

    tau: int

    def run(self) -> Tuple[List[np.float64], List[np.float64]]:
        cumulative_reward, cumulative_regret = [0], [0]

        # init
        for arm in range(len(self.mu)):
            reward, regret = self._slot(arm=arm)
            cumulative_reward.append(cumulative_reward[-1] + reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

        for t in range(len(self.mu) + 1, self.T):
            mu_hats = self.rewards / self.N
            max_mu_probs = self._softmax(mu_hats=mu_hats)
            arm = np.random.choice(len(self.mu), p=max_mu_probs)

            reward, regret = self._slot(arm=arm)
            cumulative_reward.append(cumulative_reward[-1] + reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

        return cumulative_reward, cumulative_regret

    def _softmax(self, mu_hats: np.ndarray) -> np.ndarray:
        """Softmax function.

        Args:
            mu_hats (np.ndarray): Estimated reward of each arm.

        Returns:
            np.ndarray: The probability of selection for each arm.
        """
        return np.exp(mu_hats / self.tau) / np.sum(np.exp(mu_hats / self.tau))


@dataclass
class ThompsonSampling(BaseBandit):
    """Thompson sampling bandit algorithm.

    Args:
        alpha (float): The alpha parameter of the beta prior distribution.
        beta (float): The beta parameter of the beta prior distribution.
    """

    alpha: float
    beta: float

    def run(self) -> Tuple[List[np.float64], List[np.float64]]:
        cumulative_reward, cumulative_regret = [0], [0]
        for t in range(self.T):
            mu_hats = np.random.beta(
                a=(self.alpha + self.rewards), b=(self.beta + self.N - self.rewards)
            )
            arg_max_arm = np.argmax(mu_hats)
            reward, regret = self._slot(arm=arg_max_arm)
            cumulative_reward.append(cumulative_reward[-1] + reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

        return cumulative_reward, cumulative_regret
