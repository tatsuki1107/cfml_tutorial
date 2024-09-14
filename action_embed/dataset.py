from dataclasses import dataclass
from pathlib import Path
from pathlib import PosixPath
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from obp.utils import softmax, sample_action_fast
from obp.dataset import linear_reward_function
from obp.dataset import OpenBanditDataset
from obp.dataset import BaseRealBanditDataset
from obp.dataset import BaseBanditDataset

from abstraction import AbstractionLearner
from policy import gen_eps_greedy


@dataclass
class SyntheticBanditDatasetWithActionEmbeds(BaseBanditDataset):
    n_actions: int
    dim_context: int
    n_cat_dim: int
    n_cat_per_dim: int
    latent_param_mat_dim: int
    n_unobserved_cat_dim: int = 0
    p_e_a_param_std: float = 1.0
    reward_noise: float = 1.0
    beta: float = -3.0
    random_state: int = 12345
    eps: float = 0.3

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

        self.latent_cat_param = self.random_.normal(
            size=(self.n_cat_dim, self.n_cat_per_dim, self.latent_param_mat_dim)
        )

        self.cat_dim_importance = self.random_.dirichlet(
            alpha=self.random_.uniform(size=self.n_cat_dim),
            size=1,
        )

    def obtain_batch_bandit_feedback(
        self, n_rounds: int, is_online: bool = False
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

        q_x_a = np.array(q_x_a).transpose(1, 2, 0)  # shape: (n_rounds, n_actions, n_cat_dim)
        q_x_e = np.array(q_x_e).transpose(1, 0, 2)  # shape: (n_rounds, n_cat_dim, n_cat_per_dim)

        cat_dim_importance_ = self.cat_dim_importance.reshape((1, 1, self.n_cat_dim))
        # shape: (n_rounds, n_actions)
        q_x_a = (q_x_a * cat_dim_importance_).sum(2)

        pi_b = gen_eps_greedy(q_x_a, eps=self.eps) if is_online else softmax(self.beta * q_x_a)
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
            context=context,
            action=action,
            p_e_d_a=self.p_e_d_a,
            action_context=action_context,
            reward=reward,
            pscore=pi_b[:, action],
            expected_reward=q_x_a,
            expected_reward_factual=expected_reward_factual,
            pi_b=pi_b
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, evaluation_policy: np.ndarray
    ) -> np.float64:
        return np.average(expected_reward, weights=evaluation_policy, axis=1).mean()


@dataclass
class ModifiedZOZOTOWNBanditDataset(OpenBanditDataset):
    @property
    def n_actions(self) -> int:
        return int(self.action.max() + 1)

    def pre_process(self) -> None:
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        len_list_ = self.position.max() + 1
        action_context = (
            self.item_context.drop(columns=["item_id", "item_feature_0"], axis=1)
            .apply(LabelEncoder().fit_transform)
            .values
        )
        
        action_context = np.repeat(action_context, len_list_, axis=0)
        tiled_position = np.tile(np.arange(len_list_), self.n_actions)
        self.unique_action_context = np.c_[action_context, tiled_position]
        self.action = self.position * self.n_actions + self.action
        self.action_context = self.unique_action_context[self.action]
        self.position = np.zeros_like(self.position)
        self.pscore /= self.position.max() + 1

    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> dict:
        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]

        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in [
            "action",
            "position",
            "reward",
            "pscore",
            "context",
            "action_context",
        ]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        bandit_feedback["unique_action_context"] = self.unique_action_context
        return bandit_feedback


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@dataclass
class ExtremeBanditDataset(BaseRealBanditDataset):
    n_components: int = 100
    reward_std: float = 1.0
    max_reward_noise: float = 0.2
    dataset_name: str = "EUR-Lex4K"  # EUR-Lex4K or Wiki10-31K
    random_state: int = 12345

    def __post_init__(self):
        self.data_path = Path().cwd() / "data" / self.dataset_name
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        if self.dataset_name == "EUR-Lex4K":
            self.min_label_frequency = 1
        elif self.dataset_name == "Wiki10-31K":
            self.min_label_frequency = 9
        self.random_ = check_random_state(self.random_state)

        self.load_raw_data()
        self.pre_process()

        # train a classifier to define a logging policy
        self._train_pi_b()
    
    @property
    def n_actions(self) -> int:
        return int(self.train_label.shape[1])


    def load_raw_data(self) -> None:
        self.train_feature, self.train_label = self._load_raw_data(
            file_path=self.data_path / "train.txt"
        )
        self.test_feature, self.test_label = self._load_raw_data(
            self.data_path / "test.txt"
        )

    def _load_raw_data(self, file_path: PosixPath) -> tuple[np.ndarray, ...]:
        with open(file_path, "r") as file:
            num_data, num_feature, num_label = file.readline().split()
            num_data, num_feature, num_label = (
                int(num_data),
                int(num_feature),
                int(num_label),
            )

            feature, label = [], []
            for _ in range(num_data):
                data_ = file.readline().split(" ")
                label_ = [int(x) for x in data_[0].split(",") if x != ""]
                feature_index = [int(x.split(":")[0]) for x in data_[1:]]
                feature_ = [float(x.split(":")[1]) for x in data_[1:]]

                label.append(
                    sp.csr_matrix(
                        ([1.0] * len(label_), label_, [0, len(label_)]),
                        shape=(1, num_label),
                    )
                )
                feature.append(
                    sp.csr_matrix(
                        (feature_, feature_index, [0, len(feature_)]),
                        shape=(1, num_feature),
                    )
                )

        return sp.vstack(feature).toarray(), sp.vstack(label).toarray()

    def pre_process(self) -> None:
        self.n_train, self.n_test = (
            self.train_feature.shape[0],
            self.test_feature.shape[0],
        )

        # delete some rare actions
        all_label = (
            sp.vstack([self.train_label, self.test_label]).astype(np.int8).toarray()
        )
        idx = all_label.sum(axis=0) >= self.min_label_frequency
        all_label = all_label[:, idx]

        # generate reward_noise (depends on each action)
        self.eta = self.random_.uniform(self.max_reward_noise, size=all_label.shape[1])

        self.train_label = sp.csr_matrix(
            all_label[: self.n_train], dtype=np.float32
        ).toarray()
        self.train_expected_rewards = sigmoid(
            x=(
                self.train_label * (1 - self.eta)
                + (1 - self.train_label) * (self.eta - 1)
            )
        )

        self.test_label = sp.csr_matrix(
            all_label[self.n_train :], dtype=np.float32
        ).toarray()
        self.test_expected_rewards = sigmoid(
            x=(
                self.test_label * (1 - self.eta)
                + (1 - self.test_label) * (self.eta - 1)
            )
        )

        self.train_contexts = self.sc.fit_transform(
            self.pca.fit_transform(self.train_feature)
        )
        self.test_contexts = self.sc.fit_transform(
            self.pca.fit_transform(self.test_feature)
        )

    def _train_pi_b(self, max_iter: int = 500, batch_size: int = 2000) -> None:
        idx = self.random_.choice(self.n_test, size=batch_size, replace=False)
        self.regressor = MultiOutputRegressor(
            Ridge(max_iter=max_iter, random_state=self.random_state)
        )
        self.regressor.fit(
            self.test_contexts[idx], self.test_expected_rewards[idx]
        )

    def compute_pi_b(self, contexts: np.ndarray, beta: float = 1.0) -> np.ndarray:
        q_x_a_hat = self.regressor.predict(contexts)
        pi_b = softmax(beta * q_x_a_hat)
        return pi_b

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:
        idx = self.random_.choice(self.n_train, size=n_rounds, replace=False)
        contexts = self.train_contexts[idx]

        pi_b = self.compute_pi_b(contexts)
        # a ~ \pi_b(\cdot|x)
        actions = sample_action_fast(pi_b)

        q_x_a_e_factual = self.train_expected_rewards[idx, actions]
        # r ~ p(r|x,a,e)
        rewards = self.random_.binomial(n=1, p=q_x_a_e_factual)

        return dict(
            context=contexts,
            action=actions,
            reward=rewards,
            pscore=pi_b[:, actions],
            expected_reward=self.train_expected_rewards[idx],
            pi_b=pi_b,
        )

    @staticmethod
    def calc_ground_truth_policy_value(
        expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> np.float64:
        return np.average(expected_reward, weights=action_dist, axis=1).mean()
