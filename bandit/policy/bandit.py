from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod

import numpy as np

from simulator import WebServer


@dataclass
class BasePolicy(ABC):
    """Base class for bandit algorithms.

    Args:
        mu (np.ndarray): True mean of each arm.
        T (int): Number of time steps.
    """

    n_arm: int
    T: int
    batch_size: int

    def __post_init__(self) -> None:
        """Initialize the bandit algorithm."""

        self.N = np.zeros(self.n_arm, dtype=np.int64)
        self.batched_N = np.zeros(self.n_arm, dtype=np.int64)

        self.rewards = np.zeros(self.n_arm)
        self.batched_rewards = np.zeros(self.n_arm)

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

    def _update_estimator(self) -> None:
        """Update the estimated reward of each arm."""
        self.N += self.batched_N
        self.rewards += self.batched_rewards

        self.batched_N = np.zeros(self.n_arm, dtype=np.int64)
        self.batched_rewards = np.zeros(self.n_arm)

    def _append_batch_data(
        self, arm: int, reward: np.float64, regret: np.float64
    ) -> None:
        """Update the bandit algorithm data.

        Args:
            arm (int): The arm played.
            reward (np.float64): The reward.
            regret (np.float64): The regret.
        """

        self.batched_N[arm] += 1
        self.batched_rewards[arm] += reward

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

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.batch_size > 1:
            raise ValueError("Epsilon-greedy does not support batch processing.")

    def run(self, web_server: WebServer) -> Dict[str, List[np.float64]]:

        num_search_per_arm = int(self.epsilon * (self.T / self.n_arm))
        num_utilization = self.T - int(num_search_per_arm * self.n_arm) - 1

        for t in range(num_search_per_arm):
            for arm in range(self.n_arm):
                reward, regret = web_server.response(arm=arm)
                self._append_batch_data(arm=arm, reward=reward, regret=regret)

        self._update_estimator()
        mu_hat_max_arm = np.argmax(self.mu_hats)

        for t in range(num_utilization):
            reward, regret = web_server.response(arm=mu_hat_max_arm)
            self._append_batch_data(arm=mu_hat_max_arm, reward=reward, regret=regret)

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )

    def _update_estimator(self) -> None:
        super()._update_estimator()
        self.mu_hats = self.rewards / self.N


@dataclass
class UCB(BasePolicy):
    """Upper confidence bound bandit algorithm."""

    def run(self, web_server: WebServer) -> Dict[str, List[np.float64]]:

        # init
        for arm in range(self.n_arm):
            reward, regret = web_server.response(arm=arm)
            self._append_batch_data(arm=arm, reward=reward, regret=regret)

        self._update_estimator()

        for t in range(self.n_arm + 1, self.T):
            ucb_scores = self.mu_hats + np.sqrt(np.log(t) / (2 * self.N))
            ucb_max_arm = np.argmax(ucb_scores)

            reward, regret = web_server.response(arm=ucb_max_arm)
            self._append_batch_data(arm=ucb_max_arm, reward=reward, regret=regret)

            if t % self.batch_size == 0:
                self._update_estimator()

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )

    def _update_estimator(self) -> None:
        super()._update_estimator()
        self.mu_hats = self.rewards / self.N


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
            self._append_batch_data(arm=arm, reward=reward, regret=regret)

        self._update_estimator()

        for t in range(self.n_arm + 1, self.T):
            max_mu_probs = self._softmax(mu_hats=self.mu_hats)
            arm = np.random.choice(self.n_arm, p=max_mu_probs)

            reward, regret = web_server.response(arm=arm)
            self._append_batch_data(arm=arm, reward=reward, regret=regret)

            if t % self.batch_size == 0:
                self._update_estimator()

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )

    def _update_estimator(self) -> None:
        super()._update_estimator()
        self.mu_hats = self.rewards / self.N

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

        self._update_estimator()

        for t in range(1, self.T + 1):
            arg_max_arm = np.argmax(self.mu_hats)
            reward, regret = web_server.response(arm=arg_max_arm)
            self._append_batch_data(arm=arg_max_arm, reward=reward, regret=regret)

            if t % self.batch_size == 0:
                self._update_estimator()

        cumulative_reward = self._calc_cumulative_reward()

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )

    def _update_estimator(self) -> None:
        super()._update_estimator()
        self.mu_hats = np.random.beta(
            a=(self.alpha + self.rewards), b=(self.beta + self.N - self.rewards)
        )
