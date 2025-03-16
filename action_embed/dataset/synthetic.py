from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from obp.utils import softmax, sample_action_fast
from obp.dataset import linear_behavior_policy
from obp.dataset import linear_reward_function
from obp.dataset import BaseBanditDataset
from scipy.stats import rankdata


@dataclass
class SyntheticBanditDatasetWithActionEmbeds(BaseBanditDataset):
    n_actions: int
    dim_context: int
    n_cat_dim: int
    n_cat_per_dim: int
    latent_param_mat_dim: int
    n_unobserved_cat_dim: int = 0
    p_e_a_param_std: float = 1.0
    is_probabilistic_embed: bool = True
    reward_noise: float = 1.0
    beta: float = -1.0
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self._define_action_embed()

    def _define_action_embed(self) -> None:
        # p(e_d|x,a)
        self.p_e_d_a = softmax(
            self.random_.normal(
                scale=self.p_e_a_param_std,
                size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim),
            )
        )
        if not self.is_probabilistic_embed:
            p_e_d_a_ = np.zeros((self.n_actions, self.n_cat_per_dim, self.n_cat_dim))
            p_e_d_a_[np.arange(self.n_actions)[:, None], self.p_e_d_a.argmax(1)] = 1.0
            self.p_e_d_a = p_e_d_a_

        self.latent_cat_param = self.random_.normal(
            size=(self.n_cat_dim, self.n_cat_per_dim, self.latent_param_mat_dim)
        )

        self.cat_dim_importance = self.random_.dirichlet(
            alpha=self.random_.uniform(size=self.n_cat_dim),
            size=1,
        )

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
    ) -> dict:
        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        # r
        q_x_a, q_x_e = [], []
        for d in range(self.n_cat_dim):
            q_x_e_d = linear_reward_function(
                context=context,
                action_context=self.latent_cat_param[d],
                random_state=self.random_state + d,
            )
            q_x_a_d = q_x_e_d @ self.p_e_d_a[:, :, d].T

            q_x_a.append(q_x_a_d)
            q_x_e.append(q_x_e_d)

        q_x_a = np.array(q_x_a).transpose(
            1, 2, 0
        )  # shape: (n_rounds, n_actions, n_cat_dim)
        q_x_e = np.array(q_x_e).transpose(
            1, 0, 2
        )  # shape: (n_rounds, n_cat_dim, n_cat_per_dim)

        cat_dim_importance_ = self.cat_dim_importance.reshape((1, 1, self.n_cat_dim))
        # shape: (n_rounds, n_actions)
        q_x_a = (q_x_a * cat_dim_importance_).sum(2)

        pi_b = softmax(self.beta * q_x_a)
        # a ~ \pi_b(\cdot|x)
        action = sample_action_fast(pi_b)

        action_context = []
        for d in range(self.n_cat_dim):
            # e_d ~ p(e_d|x,a), e ~ \prod_{d} p(e_d|x,a)
            action_context_d = sample_action_fast(self.p_e_d_a[action, :, d])
            action_context.append(action_context_d)

        action_context = np.array(action_context).T

        cat_dim_importance_ = self.cat_dim_importance.reshape((1, self.n_cat_dim, 1))
        expected_reward_factual = (cat_dim_importance_ * q_x_e)[
            np.arange(n_rounds)[:, None], np.arange(self.n_cat_dim), action_context
        ].sum(1)

        # r ~ p(r|x,a,e) = p(r|x,e)
        reward = self.random_.normal(
            loc=expected_reward_factual, scale=self.reward_noise
        )

        return dict(
            n_rounds=n_rounds,
            context=context,
            action=action,
            p_e_d_a=self.p_e_d_a,
            e_a=None if self.is_probabilistic_embed else self.p_e_d_a.argmax(1),
            action_context=action_context,
            reward=reward,
            pscore=pi_b[np.arange(n_rounds), action],
            expected_reward=q_x_a,
            expected_reward_factual=expected_reward_factual,
            pi_b=pi_b,
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, evaluation_policy: np.ndarray
    ) -> np.float64:
        return np.average(expected_reward, weights=evaluation_policy, axis=1).mean()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@dataclass
class SyntheticBanditDatasetWithCluster(BaseBanditDataset):
    n_users: int
    dim_context: int
    n_actions: int
    n_cat_per_dim: int
    n_cat_dim: int
    n_clusters: int
    beta: float
    reward_noise: float
    beta_user: float = 0.0
    n_deficient_actions: int = 0
    reward_type: str = "continuous"
    random_state: int = 12345

    def __post_init__(self) -> None:
        # p_u
        random_ = check_random_state(self.random_state)
        p_u_logits = random_.normal(size=self.n_users)
        self.p_u = softmax(self.beta_user * p_u_logits[None, :])[0]

        random_ = check_random_state(self.random_state)
        # x_u
        self.user_contexts = random_.normal(size=(self.n_users, self.dim_context))
        self.fixed_user_contexts = {u: c for u, c in enumerate(self.user_contexts)}

        random_ = check_random_state(self.random_state)
        # deterministic action embeddings
        self.action_contexts = random_.normal(
            size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim)
        ).argmax(1)
        self.p_e_d_a = np.zeros((self.n_actions, self.n_cat_per_dim, self.n_cat_dim))
        self.p_e_d_a[
            np.arange(self.n_actions)[:, None],
            self.action_contexts,
            np.arange(self.n_cat_dim)[None, :],
        ] = 1

        self.action_context_one_hot = OneHotEncoder(
            drop="first", sparse=False
        ).fit_transform(self.action_contexts)
        self.fixed_action_contexts = {
            a: c for a, c in enumerate(self.action_context_one_hot)
        }

        self.true_clusters = linear_behavior_policy(
            context=self.action_context_one_hot,
            action_context=np.eye(self.n_clusters),
            random_state=self.random_state,
        ).argmax(1)
        self.n_true_clusters = np.unique(self.true_clusters).shape[0]
        self.true_clusters = rankdata(-self.true_clusters, method="dense") - 1
        # context-free cluster
        self.true_clusters = np.tile(self.true_clusters, reps=(self.n_users, 1))

        self.g_x_c = linear_reward_function(
            context=self.user_contexts,
            action_context=np.eye(self.n_true_clusters),
            random_state=self.random_state,
        )
        self.g_x_c = (self.g_x_c - self.g_x_c.mean()) / self.g_x_c.std()
        self.g_x_c -= self.g_x_c.min()

        g_x_c_a = self.g_x_c[np.arange(self.n_users)[:, None], self.true_clusters]

        self.h_x_a = linear_reward_function(
            context=self.user_contexts,
            action_context=self.action_context_one_hot,
            random_state=self.random_state,
        )
        self.h_x_a = (self.h_x_a - self.h_x_a.mean()) / self.h_x_a.std()
        self.h_x_a -= self.h_x_a.min()

        # conjunct effect model (CEM)
        self.q_x_a = self.h_x_a + g_x_c_a

        # \pi_0(a|x)
        self._define_logging_policy()

        if self.reward_type == "continuous":
            # define the reward variance
            random_ = check_random_state(self.random_state)
            eps = 1e-6
            self.sigma_x_a = (
                random_.uniform(
                    0, self.reward_noise, size=(self.n_users, self.n_actions)
                )
                + eps
            )
            self.squared_q_x_a = self.sigma_x_a**2 + self.q_x_a**2

        elif self.reward_type == "binary":
            self.q_x_a = sigmoid(self.q_x_a)
            self.squared_q_x_a = self.q_x_a
            self.sigma_x_a = np.sqrt(self.q_x_a * (1 - self.q_x_a))

        else:
            raise ValueError("reward_type must be either 'continuous' or 'binary'")

        self.true_dist_dict = {
            "p_u": self.p_u,
            "x_u": self.user_contexts,
            "pi_0_a_x": self.pi_0_a_x,
            "q_x_a": self.q_x_a,
            "h_x_a": self.h_x_a,
            "g_x_c": self.g_x_c,
            "squared_q_x_a": self.squared_q_x_a,
            "phi_x_a": self.true_clusters,
            "n_clusters": self.n_true_clusters,
        }

        self.random_ = check_random_state(self.random_state)

    def _define_logging_policy(self) -> None:
        random_ = check_random_state(self.random_state)
        if self.n_deficient_actions > 0:
            self.pi_0_a_x = np.zeros_like(self.q_x_a)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                random_.gumbel(size=(self.n_users, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(self.n_users), (n_supported_actions, 1)).T,
                supported_actions,
            )
            self.pi_0_a_x[supported_actions_idx] = softmax(
                self.beta * self.q_x_a[supported_actions_idx]
            )
        else:
            self.pi_0_a_x = softmax(self.beta * self.q_x_a)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:
        # x ~ p(x)
        user_idx = self.random_.choice(self.n_users, size=n_rounds, p=self.p_u)
        context = self.user_contexts[user_idx]

        # a ~ \pi_b(\cdot|x_u)
        pi_b = self.pi_0_a_x[user_idx]
        actions = sample_action_fast(pi_b)

        phi_a = train_contextfree_cluster(
            n_actions=self.n_actions,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        learned_phi_x_a = np.tile(phi_a, reps=(self.n_users, 1))
        learned_clusters = learned_phi_x_a[user_idx, actions]

        # e ~ p(\cdot|x_u,a)
        action_contexts = self.action_contexts[actions]

        # r ~ p(r|x_u,a)
        expected_reward_factual = self.q_x_a[user_idx, actions]

        if self.reward_type == "continuous":
            reward_noise_factual = self.sigma_x_a[user_idx, actions]
            rewards = self.random_.normal(expected_reward_factual, reward_noise_factual)

        elif self.reward_type == "binary":
            rewards = self.random_.binomial(n=1, p=expected_reward_factual)

        return dict(
            n_users=self.n_users,
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            n_learned_clusters=self.n_clusters,
            user_idx=user_idx,
            context=context,
            fixed_user_context=self.fixed_user_contexts,
            action=actions,
            action_context=action_contexts,
            action_context_one_hot=self.action_context_one_hot,
            fixed_action_context=self.fixed_action_contexts,
            cluster=learned_clusters,
            reward=rewards,
            pscore=pi_b[np.arange(n_rounds), actions],
            pi_b=pi_b,
            p_e_d_a=self.p_e_d_a,
            phi_x_a=learned_phi_x_a,
            x_u=self.user_contexts,
            # unknown data
            n_true_clusters=self.n_true_clusters,
            true_phi_x_a=self.true_clusters[user_idx],
            true_cluster=self.true_clusters[user_idx, actions],
        )

    def calc_ground_truth_policy_value(self, pi_e: np.ndarray) -> np.float64:
        return (self.p_u[:, None] * pi_e * self.q_x_a).sum()


def train_contextfree_cluster(
    n_actions: int, n_clusters: int, random_state: Optional[int] = None
) -> np.ndarray:
    random_ = check_random_state(random_state)

    phi_a = random_.randint(n_clusters, size=n_actions)
    return phi_a
