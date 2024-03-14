from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """The sigmoid function.

    Args:
        x (np.ndarray): The input.

    Returns:
        np.ndarray: The output.
    """

    return np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))


@dataclass
class BaseContextualPolicy(ABC):
    """Base class for contextual bandit algorithms.

    Args:
        n_action (int): Number of actions.
        dim_context (int): Dimension of action context.
        noise_ver (float): Variance of the noise.
        noise_zero_ver (float): Variance of the noise.
    """

    n_action: int
    dim_context: int
    noise_ver: float
    noise_zero_ver: float

    @abstractmethod
    def select_action(self, user_context: np.ndarray, t: Optional[int] = None) -> int:
        pass

    @abstractmethod
    def update_parameter(self, batch_data) -> None:
        pass


@dataclass
class LinUCB(BaseContextualPolicy):
    """Linear Upper Confidence Bound (LinUCB) algorithm.

    Args:
        alpha (float): The parameter of the confidence bound.
    """

    alpha: float

    def __post_init__(self) -> None:

        self.inversed_A = np.array(
            [np.eye(self.dim_context) for _ in range(self.n_action)]
        )
        self.vector_b = np.zeros((self.n_action, self.dim_context))

        # initialize parameter
        self._calc_global_optimum()

    def select_action(self, contexts: np.ndarray, t: int) -> int:
        # contexts をもとに速攻でucbスコアを算出.
        alpha_t = self.alpha * np.sqrt(np.log(t))
        ucb_scores = self._calc_ucb_scores(
            contexts=contexts, theta_hats=self.theta_hats, alpha_t=alpha_t
        )

        # アクションを選択して、クライアントに返す
        selected_action = np.argmax(ucb_scores)

        return selected_action

    def _calc_global_optimum(self) -> None:
        """Estimate the least squares estimator

        Returns:
            np.ndarray: The least squares estimator.
        """

        theta_hats = []
        for action in range(self.n_action):
            theta_hat = np.dot(self.inversed_A[action], self.vector_b[action]).tolist()
            theta_hats.append(theta_hat)

        self.theta_hats = np.array(theta_hats)

    def _calc_ucb_scores(
        self, contexts: np.ndarray, theta_hats: np.ndarray, alpha_t: np.float64
    ) -> list:
        """Calculate the upper confidence bound scores.

        Args:
            contexts (np.ndarray): The contexts.
            theta_hat (np.ndarray): The least squares estimator.
            alpha_t (np.float64): The parameter of the confidence bound.

        Returns:
            list: The upper confidence bound scores.
        """

        ucb_scores = []
        for action in range(self.n_action):
            ucb_score = np.dot(
                contexts[action], theta_hats[action]
            ) + alpha_t * np.sqrt(
                self.noise_ver
                * (contexts[action].T @ self.inversed_A[action] @ contexts[action])
            )
            ucb_scores.append(ucb_score)

        return ucb_scores

    def update_parameter(self, batch_data: list) -> None:
        for context, action, reward, _ in batch_data:
            self._update_inversed_A(context, action)
            self._update_vector_b(context, action, reward)

        self._calc_global_optimum()

    def _update_inversed_A(
        self, context: np.ndarray, selected_action: np.int64
    ) -> None:
        """Update the inverse of the design matrix A.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
        """
        molecule = (
            self.inversed_A[selected_action]
            @ np.outer(context, context)
            @ self.inversed_A[selected_action]
        )
        denominator = 1 + (context.T @ self.inversed_A[selected_action] @ context)
        self.inversed_A[selected_action] -= molecule / denominator

    def _update_vector_b(
        self, context: np.ndarray, selected_action: np.int64, reward: np.float64
    ) -> None:
        """Update the vector b.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
            reward (np.float64): The reward.
        """
        self.vector_b[selected_action] += context * reward


@dataclass
class LinThompsonSampling(BaseContextualPolicy):
    """Linear Thompson Sampling algorithm."""

    def __post_init__(self) -> None:
        """Initialize the inversed_A and vector_b."""

        self.inversed_A = np.array(
            [np.eye(self.dim_context) for _ in range(self.n_action)]
        )
        self.vector_b = np.zeros((self.n_action, self.dim_context))

    def select_action(self, contexts: np.ndarray, t: Optional[int] = None) -> int:
        # 各アクションに対してパラメータをサンプリング
        self.theta_hats = self._sampling_theta_hats()
        # 線形バンディットの報酬モデルから各アクションに対する報酬期待値を計算
        expected_rewards = self._get_expected_rewards(
            contexts=contexts, theta_hats=self.theta_hats
        )

        # 期待値最大のアクションを選択、
        selected_action = np.argmax(expected_rewards)

        return selected_action

    def _sampling_theta_hats(self) -> np.ndarray:
        """Sampling from a multivariate normal distribution.

        Returns:
            np.ndarray: The sampled parameters.
        """

        theta_hats = []
        for action in range(self.n_action):
            theta_hat = np.random.multivariate_normal(
                mean=np.dot(self.inversed_A[action], self.vector_b[action]),
                cov=self.noise_ver * self.inversed_A[action],
            )
            theta_hats.append(theta_hat)

        return np.array(theta_hats)

    def _get_expected_rewards(
        self, contexts: np.ndarray, theta_hats: np.ndarray
    ) -> np.ndarray:
        """Get the expected rewards.

        Args:
            contexts (np.ndarray): The contexts.
            theta_hats (np.ndarray): The sampled parameters.

        Returns:
            np.ndarray: The expected rewards.
        """

        expected_rewards = []
        for action in range(self.n_action):
            expected_reward = np.dot(contexts[action], theta_hats[action])
            expected_rewards.append(expected_reward)

        return np.array(expected_rewards)

    def update_parameter(self, batch_data: list) -> None:
        for context, action, reward, _ in batch_data:
            self._update_inversed_A(context, action)
            self._update_vector_b(context, action, reward)

    def _update_inversed_A(
        self, context: np.ndarray, selected_action: np.int64
    ) -> None:
        """Update the inverse of the design matrix A.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
        """
        molecule = (
            self.inversed_A[selected_action]
            @ np.outer(context, context)
            @ self.inversed_A[selected_action]
        )
        denominator = 1 + (context.T @ self.inversed_A[selected_action] @ context)
        self.inversed_A[selected_action] -= molecule / denominator

    def _update_vector_b(
        self, context: np.ndarray, selected_action: np.int64, reward: np.float64
    ) -> None:
        """Update the vector b.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
            reward (np.float64): The reward.
        """
        self.vector_b[selected_action] += context * reward


@dataclass
class LogisticThompsonSampling(BaseContextualPolicy):
    n_epoch: int = 1

    def __post_init__(self) -> None:

        self.theta_hats = np.zeros((self.n_action, self.dim_context))
        inversed_noise_ver = 1 / self.noise_zero_ver
        self.inversed_hessian = inversed_noise_ver * np.array(
            [
                np.linalg.inv(inversed_noise_ver * np.eye(self.dim_context))
                for _ in range(self.n_action)
            ]
        )

        self.logged_contexts = [[] for _ in range(self.n_action)]
        self.logged_rewards = [[] for _ in range(self.n_action)]

    def select_action(self, contexts: np.ndarray, t: Optional[int] = None) -> int:

        theta_tildes = self._sampling_theta_tilde()

        # 線形バンディットの報酬モデルから各アクションに対する報酬期待値を計算
        expected_rewards = self._get_expected_rewards(
            contexts=contexts, theta_tildes=theta_tildes
        )

        # 期待値最大のアクションを選択、
        selected_action = np.argmax(expected_rewards)

        return selected_action

    def _sampling_theta_tilde(self) -> np.ndarray:
        theta_tildes = []
        for action in range(self.n_action):
            theta_tilde = np.random.multivariate_normal(
                mean=self.theta_hats[action],
                cov=self.inversed_hessian[action],
            )
            theta_tildes.append(theta_tilde)

        return np.array(theta_tildes)

    def _get_expected_rewards(
        self, contexts: np.ndarray, theta_tildes: np.ndarray
    ) -> np.ndarray:
        """Get the expected rewards.

        Args:
            contexts (np.ndarray): The contexts.
            theta_hats (np.ndarray): The sampled parameters.

        Returns:
            np.ndarray: The expected rewards.
        """

        expected_rewards = []
        for action in range(self.n_action):
            expected_reward = np.dot(contexts[action], theta_tildes[action])
            expected_rewards.append(expected_reward)

        return np.array(expected_rewards)

    def update_parameter(self, batch_data: list) -> None:

        batched_unique_actions = set()
        for context, action, reward, _ in batch_data:
            self.logged_contexts[action].append(context.tolist())
            self.logged_rewards[action].append(reward)
            batched_unique_actions.add(action)

        for action in batched_unique_actions:
            self._update_theta_hats(action)

    def _update_theta_hats(self, action: np.int64) -> None:

        contexts = np.array(self.logged_contexts[action])
        rewards = np.array(self.logged_rewards[action])

        for _ in range(self.n_epoch):
            gradient = self._calc_gradient(contexts, action, rewards)
            self.inversed_hessian[action] = self._calc_inversed_hessian(
                contexts, action
            )

            self.theta_hats[action] = self.theta_hats[action] - np.dot(
                self.inversed_hessian[action], gradient
            )

    def _calc_gradient(
        self, contexts: np.ndarray, action: np.int64, rewards: np.float64
    ) -> np.ndarray:
        mu_hats = sigmoid(np.dot(contexts, self.theta_hats[action]))
        gradient = (1 / self.noise_zero_ver) * self.theta_hats[action]
        gradient += np.sum((mu_hats - rewards)[:, None] * contexts, axis=0)

        return gradient

    def _calc_inversed_hessian(
        self, contexts: np.ndarray, action: np.int64
    ) -> np.ndarray:
        mu_hats = sigmoid(np.dot(contexts, self.theta_hats[action]))
        hessian = (1 / self.noise_zero_ver) * np.eye(self.dim_context)
        for t in range(contexts.shape[0]):
            hessian += (
                mu_hats[t] * (1 - mu_hats[t]) * np.outer(contexts[t], contexts[t].T)
            )

        return np.linalg.inv(hessian)
