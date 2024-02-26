from dataclasses import dataclass
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class WebServer:
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
class BasePolicy(ABC):
    """Base class for bandit algorithms.

    Args:
        mu (np.ndarray): True mean of each arm.
        T (int): Number of time steps.
    """

    n_arm: int
    T: int

    def __post_init__(self) -> None:
        """Initialize the bandit algorithm."""

        self.N = np.zeros(self.n_arm, dtype=np.int64)
        self.rewards = np.zeros(self.n_arm)

        self.reward_per_time = []
        self.cumulative_regret = [0]

    @abstractmethod
    def run(self, *kwargs) -> Dict[str, List[np.float64]]:
        """Run the bandit algorithm.
        Returns:
            Tuple[List[np.float64], List[np.float64]]:
            Cumulative reward and cumulative regret.
        """
        pass

    def _update_data(self, arm: int, reward: np.float64, regret: np.float64) -> None:
        """Update the bandit algorithm data.

        Args:
            arm (int): The arm played.
            reward (np.float64): The reward.
            regret (np.float64): The regret.
        """
        self.N[arm] += 1
        self.rewards[arm] += reward
        self.reward_per_time.append(reward)
        self.cumulative_regret.append(self.cumulative_regret[-1] + regret)

    def _calc_cumulative_reward(self) -> List[np.float64]:
        """Calculate the cumulative reward.

        Returns:
            List[np.float64]: The cumulative reward.
        """

        cumulative_reward = [0]
        for i in range(len(self.reward_per_time)):
            curr_sum = cumulative_reward[-1] + self.reward_per_time[i]
            cumulative_reward.append(curr_sum)

        return cumulative_reward


@dataclass
class EpsilonGreedy(BasePolicy):
    """Epsilon-greedy bandit algorithm.

    Args:
        epsilon (float): The exploration rate.
    """

    epsilon: float

    def run(self, web_server: WebServer) -> Dict[str, List[np.float64]]:

        num_search_per_arm = int(self.epsilon * (self.T / self.n_arm))
        num_utilization = self.T - int(num_search_per_arm * self.n_arm) - 1

        for t in range(num_search_per_arm):
            for arm in range(self.n_arm):
                reward, regret = web_server.response(arm=arm)
                self._update_data(arm=arm, reward=reward, regret=regret)

        mu_hats = self.rewards / self.N
        mu_hat_max_arm = np.argmax(mu_hats)

        for t in range(num_utilization):
            reward, regret = web_server.response(arm=mu_hat_max_arm)
            self._update_data(arm=mu_hat_max_arm, reward=reward, regret=regret)

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )


@dataclass
class UCB(BasePolicy):
    """Upper confidence bound bandit algorithm."""

    def run(self, web_server: WebServer) -> Dict[str, List[np.float64]]:

        # init
        for arm in range(self.n_arm):
            reward, regret = web_server.response(arm=arm)
            self._update_data(arm=arm, reward=reward, regret=regret)

        for t in range(self.n_arm + 1, self.T):
            mu_hats = self.rewards / self.N
            ucb_scores = mu_hats + np.sqrt(np.log(t) / (2 * self.N))
            ucb_max_arm = np.argmax(ucb_scores)

            reward, regret = web_server.response(arm=ucb_max_arm)
            self._update_data(arm=ucb_max_arm, reward=reward, regret=regret)

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )


@dataclass
class Softmax(BasePolicy):
    """Softmax bandit algorithm.

    Args:
        tau (int): The temperature parameter.
    """

    tau: int

    def run(self, web_server: WebServer) -> Dict[str, List[np.float64]]:

        # init
        for arm in range(self.n_arm):
            reward, regret = web_server.response(arm=arm)
            self._update_data(arm=arm, reward=reward, regret=regret)

        for t in range(self.n_arm + 1, self.T):
            mu_hats = self.rewards / self.N
            max_mu_probs = self._softmax(mu_hats=mu_hats)
            arm = np.random.choice(self.n_arm, p=max_mu_probs)

            reward, regret = web_server.response(arm=arm)
            self._update_data(arm=arm, reward=reward, regret=regret)

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )

    def _softmax(self, mu_hats: np.ndarray) -> np.ndarray:
        """Softmax function.

        Args:
            mu_hats (np.ndarray): Estimated reward of each arm.

        Returns:
            np.ndarray: The probability of selection for each arm.
        """
        return np.exp(mu_hats / self.tau) / np.sum(np.exp(mu_hats / self.tau))


@dataclass
class ThompsonSampling(BasePolicy):
    """Thompson sampling bandit algorithm.

    Args:
        alpha (float): The alpha parameter of the beta prior distribution.
        beta (float): The beta parameter of the beta prior distribution.
    """

    alpha: float
    beta: float

    def run(self, web_server: WebServer) -> Dict[str, List[np.float64]]:

        for t in range(self.T):
            mu_hats = np.random.beta(
                a=(self.alpha + self.rewards), b=(self.beta + self.N - self.rewards)
            )
            arg_max_arm = np.argmax(mu_hats)
            reward, regret = web_server.response(arm=arg_max_arm)
            self._update_data(arm=arg_max_arm, reward=reward, regret=regret)

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )
