from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

from web_server import RecommendWebServer


@dataclass
class BaseMFPolicy(ABC):
    n_action: int
    n_user: int
    dim_context: int
    T: int
    noise_ver: float
    noise_u_ver: float
    noise_a_ver: float
    batch_size: int

    def __post_init__(self):
        self.reward_per_time = []
        self.cumulative_regret = [0]

        # [(user, user_embedding, action, action_embedding ,reward), ...]
        self.batched_data = []

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def _append_batch_data(
        self,
        user: int,
        user_embedding: np.ndarray,
        action: int,
        action_embedding: np.ndarray,
        reward: np.float64,
        regret: np.float64,
    ) -> None:
        self.batched_data.append(
            (user, user_embedding, action, action_embedding, reward)
        )

        self.reward_per_time.append(reward)
        self.cumulative_regret.append(self.cumulative_regret[-1] + regret)

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
class MFThompsonSampling(BaseMFPolicy):
    """Multi-armed bandit with Thompson Sampling and Matrix Factorizarion.

    Args:
        BaseMFPolicy (_type_): _description_

    References:

    Xiaoxue Zhao, Weinan Zhang, Jun Wang.
    "Interactive Collaborative Filtering". 2013.
    """

    def __post_init__(self):
        super().__post_init__()

        self.inversed_A_u = np.array(
            [np.eye(self.dim_context) for _ in range(self.n_user)]
        )
        self.inversed_A_a = np.array(
            [np.eye(self.dim_context) for _ in range(self.n_action)]
        )
        self.vector_b_u = np.zeros((self.n_user, self.dim_context))
        self.vector_b_a = np.zeros((self.n_action, self.dim_context))

    def run(self, web_server: RecommendWebServer) -> Dict[str, List[np.float64]]:

        for t in range(self.T):
            user = web_server.request(t)
            user_embedding = self._sampling_user_embedding(user)
            action_embeddings = self._sampling_action_embedding()
            expected_rewards = self._calc_expected_rewards(
                user_embedding, action_embeddings
            )

            selected_action = np.argmax(expected_rewards)
            reward, regret = web_server.response(user, selected_action)

            self._append_batch_data(
                user=user,
                user_embedding=user_embedding,
                action=selected_action,
                action_embedding=action_embeddings[selected_action],
                reward=reward,
                regret=regret,
            )

            if t % self.batch_size == 0:
                self._update_parameter()

        cumulative_reward = self._calc_cumulative_reward(self.reward_per_time)

        return dict(
            cumulative_reward=cumulative_reward,
            reward_per_time=self.reward_per_time,
            cumulative_regret=self.cumulative_regret,
        )

    def _sampling_user_embedding(self, user: int):

        user_embedding = np.random.multivariate_normal(
            mean=np.dot(self.inversed_A_u[user], self.vector_b_u[user]),
            cov=self.noise_ver * self.inversed_A_u[user],
        )
        return user_embedding

    def _sampling_action_embedding(self):

        action_embeddings = []
        for action in range(self.n_action):
            action_embedding = np.random.multivariate_normal(
                mean=np.dot(self.inversed_A_a[action], self.vector_b_a[action]),
                cov=self.noise_ver * self.inversed_A_a[action],
            )
            action_embeddings.append(action_embedding.tolist())

        return np.array(action_embeddings)

    def _calc_expected_rewards(self, user_embedding, action_embeddings):

        expected_rewards = np.dot(action_embeddings, user_embedding)
        return expected_rewards

    def _update_parameter(self) -> None:

        for user, user_embedding, action, action_embedding, reward in self.batched_data:
            # update user parameter
            self._update_inversed_A_u(user, action_embedding)
            self._update_vector_b_u(user, action_embedding, reward)

            # update action parameter
            self._update_inversed_A_a(action, user_embedding)
            self._update_vector_b_a(action, user_embedding, reward)

        self.batched_data = []

    def _update_inversed_A_u(
        self, user: np.int64, action_embedding: np.ndarray
    ) -> None:
        """Update the inverse of the design matrix A.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
        """
        molecule = (
            self.inversed_A_u[user]
            @ np.outer(action_embedding, action_embedding)
            @ self.inversed_A_u[user]
        )
        denominator = 1 + (
            action_embedding.T @ self.inversed_A_u[user] @ action_embedding
        )
        self.inversed_A_u[user] -= molecule / denominator

    def _update_inversed_A_a(
        self, action: np.int64, user_embedding: np.ndarray
    ) -> None:
        """Update the inverse of the design matrix A.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
        """
        molecule = (
            self.inversed_A_a[action]
            @ np.outer(user_embedding, user_embedding)
            @ self.inversed_A_a[action]
        )
        denominator = 1 + (
            user_embedding.T @ self.inversed_A_a[action] @ user_embedding
        )
        self.inversed_A_a[action] -= molecule / denominator

    def _update_vector_b_u(
        self, user: np.int64, action_embedding: np.ndarray, reward: np.float64
    ) -> None:
        """Update the vector b.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
            reward (np.float64): The reward.
        """
        self.vector_b_u[user] += action_embedding * reward

    def _update_vector_b_a(
        self, action: np.int64, user_embedding: np.ndarray, reward: np.float64
    ) -> None:
        """Update the vector b.

        Args:
            context (np.ndarray): The context.
            selected_action (np.int64): The selected action.
            reward (np.float64): The reward.
        """
        self.vector_b_a[action] += user_embedding * reward
