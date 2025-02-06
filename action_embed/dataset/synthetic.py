from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from obp.utils import softmax, sample_action_fast
from obp.dataset import linear_behavior_policy
from obp.dataset import linear_reward_function
from obp.dataset import polynomial_reward_function
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
            p_e_d_a_[np.arange(self.n_actions)[:, None], self.p_e_d_a.argmax(1)] = 1.
            self.p_e_d_a = p_e_d_a_

        self.latent_cat_param = self.random_.normal(
            size=(self.n_cat_dim, self.n_cat_per_dim, self.latent_param_mat_dim)
        )

        self.cat_dim_importance = self.random_.dirichlet(
            alpha=self.random_.uniform(size=self.n_cat_dim),
            size=1,
        )

    def obtain_batch_bandit_feedback(
        self, n_rounds: int,
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


def cluster_effect_function(
    context: np.ndarray,
    cluster_context: np.ndarray,
    random_state: int,
) -> np.ndarray:
    g_x_e = polynomial_reward_function(
        context=context,
        action_context=cluster_context,
        random_state=random_state,
    )
    random_ = check_random_state(random_state)
    (a, b, c, d) = random_.uniform(-3, 3, size=4)
    x_a = 1 / context[:, :3].mean(axis=1)
    x_b = 1 / context[:, 2:8].mean(axis=1)
    x_c = context[:, 1:3].mean(axis=1)
    x_d = context[:, 5:].mean(axis=1)
    g_x_e += a * (x_a[:, np.newaxis] < 1.5)
    g_x_e += b * (x_b[:, np.newaxis] < -0.5)
    g_x_e += c * (x_c[:, np.newaxis] > 3.0)
    g_x_e += d * (x_d[:, np.newaxis] < 1.0)

    return g_x_e


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
    n_deficient_actions: int = 0
    random_state: int = 12345
    
    def __post_init__(self) -> None:
        
        random_ = check_random_state(self.random_state)
        self.user_contexts = random_.normal(size=(self.n_users, self.dim_context))
        self.fixed_user_contexts = {u: c for u, c in enumerate(self.user_contexts)}

        random_ = check_random_state(self.random_state)
        # deterministic action embeddings
        self.action_contexts = random_.normal(size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim)).argmax(1)
        
        self.action_context_one_hot = OneHotEncoder(drop="first", sparse=False).fit_transform(self.action_contexts)
        self.fixed_action_contexts = {a: c for a, c in enumerate(self.action_context_one_hot)}
        
        self.clusters = linear_behavior_policy(
            context=self.action_context_one_hot,
            action_context=np.eye(self.n_clusters),
            random_state=self.random_state
        ).argmax(1)
        self.n_clusters = np.unique(self.clusters).shape[0]
        self.clusters = rankdata(-self.clusters, method="dense") - 1
        
        self.cluster_one_hot = np.zeros((self.n_actions, self.n_clusters))
        self.cluster_one_hot[np.arange(self.n_actions), self.clusters] = 1
        self.fixed_cluster_contexts = {a: c for a, c in enumerate(self.cluster_one_hot)}
        
        g_x_c = cluster_effect_function(
            context=self.user_contexts,
            cluster_context=np.eye(self.n_clusters),
            random_state=self.random_state,
        )
        g_x_c_a = g_x_c[:, self.clusters]
        
        h_x_a = linear_reward_function(
            context=self.user_contexts,
            action_context=self.action_context_one_hot,
            random_state=self.random_state
        )
        # conjunct effect model (CEM)
        self.q_x_a = h_x_a + g_x_c_a
        
        self.random_ = check_random_state(self.random_state)
    
    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:
        
        # x ~ p(x)
        user_idx = self.random_.choice(self.n_users, size=n_rounds)
        context = self.user_contexts[user_idx]
        
        q_x_a = self.q_x_a[user_idx]
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(q_x_a)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * q_x_a[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * q_x_a)
        
        # a ~ \pi_b(\cdot|x)
        actions = sample_action_fast(pi_b)
        
        # e ~ p(\cdot|x,a)
        action_contexts = self.action_contexts[actions]
        
        expected_reward_factual = q_x_a[np.arange(n_rounds), actions]
        rewards = self.random_.normal(expected_reward_factual, self.reward_noise)
        
        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            n_clusters=self.n_clusters,
            user_idx=user_idx,
            context=context,
            fixed_user_context=self.fixed_user_contexts,
            action=actions,
            unique_action_context=self.action_contexts,
            action_context=action_contexts,
            action_context_one_hot=self.action_context_one_hot,
            fixed_action_context=self.fixed_action_contexts,
            cluster=self.clusters,
            cluster_one_hot=self.cluster_one_hot,
            fixed_cluster_context=self.fixed_cluster_contexts,
            reward=rewards,
            pscore=pi_b[np.arange(n_rounds), actions],
            expected_reward=self.q_x_a,
            pi_b=pi_b,
        )
    
    def calc_ground_truth_policy_value(self, q_x_a: np.ndarray, pi_e: np.ndarray) -> np.float64:
        return np.average(q_x_a, weights=pi_e, axis=1).mean()
