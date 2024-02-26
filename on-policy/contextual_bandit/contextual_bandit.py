from dataclasses import dataclass
from typing import List, Tuple, Dict
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
class ContextualWebServer:
    n_action: int
    dim_action_context: int
    dim_user_context: int
    dim_context: int
    reward_type: str  # "binary" or "continuous"
    noise_ver: float
    seed: int

    def __post_init__(self) -> None:
        """Initialize the action contexts and the parameters."""
        # i.i.d
        np.random.seed(self.seed)
        self.action_contexts = np.random.normal(
            0, 1, size=(self.n_action, self.dim_action_context)
        )

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


@dataclass
class BaseContextualPolicy(ABC):
    """Base class for contextual bandit algorithms.

    Args:
        T (int): Number of time steps.
        n_action (int): Number of actions.
        dim_action_context (int): Dimension of action context.
        dim_user_context (int): Dimension of user context.
        noise_ver (float): Variance of the noise.
        seed (int): Random seed.
    """

    T: int
    n_action: int
    dim_action_context: int
    dim_context: int
    seed: int
    noise_ver: float
    noise_zero_ver: float

    def __post_init__(self) -> None:

        np.random.seed(self.seed)
        self.action_contexts = np.random.normal(
            0, 1, size=(self.n_action, self.dim_action_context)
        )

    @abstractmethod
    def run(self, *kwargs) -> Dict[str, List[np.float64]]:
        """Run the contextual bandit algorithm.

        Returns:
            Tuple[List[np.float64], List[np.float64]]:
            Cumulative reward and cumulative regret.
        """
        pass

    def _preprocess_contexts(self, user_context: np.ndarray) -> np.ndarray:
        contexts = []
        for action_context in self.action_contexts:
            interaction_vector = np.array(
                [u * a for a in action_context for u in user_context]
            )
            context = np.r_[action_context, user_context, interaction_vector].tolist()
            contexts.append(context)

        return np.array(contexts)

    def _calc_cumulative_reward(self, reward_per_time) -> List[np.float64]:
        """Calculate the cumulative reward.

        Returns:
            List[np.float64]: The cumulative reward.
        """

        cumulative_reward = [0]
        for i in range(len(reward_per_time)):
            curr_sum = cumulative_reward[-1] + reward_per_time[i]
            cumulative_reward.append(curr_sum)

        return cumulative_reward


@dataclass
class LinUCB(BaseContextualPolicy):
    """Linear Upper Confidence Bound (LinUCB) algorithm.

    Args:
        alpha (float): The parameter of the confidence bound.
    """

    alpha: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inversed_A = np.array(
            [np.eye(self.dim_context) for _ in range(self.n_action)]
        )
        self.vector_b = np.zeros((self.n_action, self.dim_context))

    def run(
        self, web_server: ContextualWebServer
    ) -> Tuple[List[np.float64], List[np.float64]]:

        reward_per_time, cumulative_regret = [], [0]
        for t in range(1, self.T + 1):
            # 時刻t時点でweb上に訪れたユーザー文脈をサーバー側で取得
            user_context = web_server.request(t)
            # アクションの文脈との交互作用を考慮した文脈 "contexts" を生成
            contexts = self._preprocess_contexts(user_context)

            # contexts をもとに速攻でucbスコアを算出.
            theta_hat = self._get_global_optimum()
            alpha_t = self.alpha * np.sqrt(np.log(t))
            ucb_scores = self._calc_ucb_scores(
                contexts=contexts, theta_hat=theta_hat, alpha_t=alpha_t
            )

            # アクションを選択して、クライアントに返す
            selected_action = np.argmax(ucb_scores)
            # すぐに報酬がサーバー側に返ってくる
            reward, regret = web_server.response(
                contexts=contexts, selected_action=selected_action
            )

            reward_per_time.append(reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

            # update parametar
            self._update_inversed_A(
                context=contexts[selected_action], selected_action=selected_action
            )
            self._update_vector_b(
                context=contexts[selected_action],
                selected_action=selected_action,
                reward=reward,
            )

        cumulative_reward = self._calc_cumulative_reward(reward_per_time)

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=reward_per_time,
            cumulative_regret=cumulative_regret,
        )

    def _get_global_optimum(self) -> np.ndarray:
        """Estimate the least squares estimator

        Returns:
            np.ndarray: The least squares estimator.
        """

        theta_hats = []
        for action in range(self.n_action):
            theta_hat = np.dot(self.inversed_A[action], self.vector_b[action]).tolist()
            theta_hats.append(theta_hat)

        return np.array(theta_hats)

    def _calc_ucb_scores(
        self, contexts: np.ndarray, theta_hat: np.ndarray, alpha_t: np.float64
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
            ucb_score = np.dot(contexts[action], theta_hat[action]) + alpha_t * np.sqrt(
                self.noise_ver
                * (contexts[action].T @ self.inversed_A[action] @ contexts[action])
            )
            ucb_scores.append(ucb_score)

        return ucb_scores

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

        super().__post_init__()

        self.inversed_A = np.array(
            [np.eye(self.dim_context) for _ in range(self.n_action)]
        )
        self.vector_b = np.zeros((self.n_action, self.dim_context))

    def run(
        self, web_server: ContextualWebServer
    ) -> Tuple[List[np.float64], List[np.float64]]:

        reward_per_time, cumulative_regret = [], [0]
        for t in range(1, self.T + 1):
            # 時刻t時点でweb上に訪れた単一ユーザー文脈をサーバー側で取得
            user_context = web_server.request(t)
            # 各アクションの文脈との交互作用を考慮した文脈 "contexts" を生成
            contexts = self._preprocess_contexts(user_context)

            # "contexts" をもとに速攻でパラメータをベイズ推論
            theta_hats = self._sampling_theta_hats()
            # 線形バンディットの報酬モデルから各アクションに対する報酬期待値を計算
            expected_rewards = self._get_expected_rewards(
                contexts=contexts, theta_hats=theta_hats
            )

            # 期待値最大のアクションを選択して、
            selected_action = np.argmax(expected_rewards)
            # クライアントに返す. すぐに報酬がサーバー側に返ってくる
            reward, regret = web_server.response(
                contexts=contexts, selected_action=selected_action
            )

            reward_per_time.append(reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

            # update parametar
            self._update_inversed_A(
                context=contexts[selected_action], selected_action=selected_action
            )
            self._update_vector_b(
                context=contexts[selected_action],
                selected_action=selected_action,
                reward=reward,
            )

        cumulative_reward = self._calc_cumulative_reward(reward_per_time)

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=reward_per_time,
            cumulative_regret=cumulative_regret,
        )

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
        super().__post_init__()

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

    def run(
        self, web_server: ContextualWebServer
    ) -> Tuple[List[np.float64], List[np.float64]]:

        reward_per_time, cumulative_regret = [], [0]
        for t in range(1, self.T + 1):

            # 時刻t時点でweb上に訪れた単一ユーザー文脈をサーバー側で取得
            user_context = web_server.request(t)
            # 各アクションの文脈との交互作用を考慮した文脈 "contexts" を生成
            contexts = self._preprocess_contexts(user_context)

            theta_tildes = self._sampling_theta_tilde()

            # 線形バンディットの報酬モデルから各アクションに対する報酬期待値を計算
            expected_rewards = self._get_expected_rewards(
                contexts=contexts, theta_tildes=theta_tildes
            )

            # 期待値最大のアクションを選択して、
            selected_action = np.argmax(expected_rewards)
            # クライアントに返す. すぐに報酬がサーバー側に返ってくる
            reward, regret = web_server.response(
                contexts=contexts, selected_action=selected_action
            )

            reward_per_time.append(reward)
            cumulative_regret.append(cumulative_regret[-1] + regret)

            self._update_theta_hats(
                context=contexts[selected_action],
                selected_action=selected_action,
                reward=reward,
            )

        cumulative_reward = self._calc_cumulative_reward(reward_per_time)

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=reward_per_time,
            cumulative_regret=cumulative_regret,
        )

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

    def _update_theta_hats(
        self, context: np.ndarray, selected_action: np.int64, reward: np.float64
    ) -> None:

        self.logged_contexts[selected_action].append(context.tolist())
        self.logged_rewards[selected_action].append(reward)

        contexts = np.array(self.logged_contexts[selected_action])
        rewards = np.array(self.logged_rewards[selected_action])

        for _ in range(self.n_epoch):
            gradient = self._calc_gradient(contexts, selected_action, rewards)
            self.inversed_hessian[selected_action] = self._calc_inversed_hessian(
                contexts, selected_action
            )

            self.theta_hats[selected_action] = self.theta_hats[
                selected_action
            ] - np.dot(self.inversed_hessian[selected_action], gradient)

    def _calc_gradient(
        self, contexts: np.ndarray, selected_action: np.int64, rewards: np.float64
    ) -> np.ndarray:
        mu_hats = sigmoid(np.dot(contexts, self.theta_hats[selected_action]))
        gradient = (1 / self.noise_zero_ver) * self.theta_hats[selected_action]
        gradient += np.sum((mu_hats - rewards)[:, None] * contexts, axis=0)

        return gradient

    def _calc_inversed_hessian(
        self, contexts: np.ndarray, selected_action: np.int64
    ) -> np.ndarray:
        mu_hats = sigmoid(np.dot(contexts, self.theta_hats[selected_action]))
        hessian = (1 / self.noise_zero_ver) * np.eye(self.dim_context)
        for t in range(contexts.shape[0]):
            hessian += (
                mu_hats[t] * (1 - mu_hats[t]) * np.outer(contexts[t], contexts[t].T)
            )

        return np.linalg.inv(hessian)
