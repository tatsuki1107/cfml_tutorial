from dataclasses import dataclass
from pathlib import Path
from pathlib import PosixPath
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from obp.utils import softmax, sample_action_fast
from obp.dataset import OpenBanditDataset
from obp.dataset import BaseRealBanditDataset


@dataclass
class ModifiedZOZOTOWNBanditDataset(OpenBanditDataset):
    n_clusters: int = 30
    random_state: int = 12345
    
    @property
    def n_actions(self) -> int:
        return self.action.max() + 1

    def pre_process(self) -> None:
        user_cols = self.data.columns.str.contains("user_feature")
        context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        self.user_context = np.unique(context, axis=0)
        self.n_users = self.user_context.shape[0]
        
        self.context = np.zeros(len(context), dtype=int)
        for u, user_context_ in enumerate(self.user_context):
            user_mask = np.all(user_context_ == context, axis=1)
            self.context[user_mask] = u
        
        self.fixed_user_context = {u: c for u, c in enumerate(self.user_context)}
        
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
        self.action_context_one_hot = OneHotEncoder(
            drop="first", sparse=False
        ).fit_transform(self.action_context)
        self.fixed_action_contexts = {
            a: c for a, c in enumerate(self.action_context_one_hot)
        }
        
        # context-free cluster
        self.cluster = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        ).fit_predict(self.unique_action_context)
        self.cluster = np.tile(self.cluster, reps=(self.n_users, 1))
        # note that the cluster is deterministic
        n_actions_all = len(self.unique_action_context)
        self.p_c_x_a = np.zeros((self.n_users, self.n_clusters, n_actions_all))
        self.p_c_x_a[
            np.arange(self.n_users)[:, None],
            self.cluster,
            np.arange(n_actions_all)[None, :],
        ] = 1
        
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
        for key_ in ["action", "position", "reward", "pscore", "action_context"]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]

        bandit_feedback["n_rounds"] = sample_size
        user_idx = bandit_feedback["context"][bootstrap_idx]
        bandit_feedback["user_idx"] = user_idx
        bandit_feedback["context"] = self.user_context[user_idx]
        bandit_feedback["dim_context"] = self.user_context.shape[1]
        bandit_feedback["fixed_user_context"] = self.fixed_user_context
        bandit_feedback["unique_action_context"] = self.unique_action_context
        bandit_feedback["action_context_one_hot"] = self.action_context_one_hot
        bandit_feedback["fixed_action_context"] = self.fixed_action_contexts
        bandit_feedback["cluster"] = self.cluster[user_idx, bandit_feedback["action"]]
        bandit_feedback["phi_x_a"] = self.cluster[user_idx]
        bandit_feedback["p_c_x_a"] = self.p_c_x_a[user_idx]
        
        return bandit_feedback


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@dataclass
class ExtremeBanditDataset(BaseRealBanditDataset):
    n_components: int = 100
    reward_std: float = 3.0
    max_reward_noise: float = 0.2
    dataset_name: str = "EUR-Lex4K"  # EUR-Lex4K or "RCV1-2K"
    n_clusters: int = 30
    random_state: int = 12345

    def __post_init__(self):
        self.data_path = Path().cwd() / "data" / self.dataset_name
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        if self.dataset_name == "EUR-Lex4K":
            self.min_label_frequency = 1
            self.train_size, self.test_size = None, None
        elif self.dataset_name == "RCV1-2K":
            self.min_label_frequency = 1
            self.train_size, self.test_size = 15000, 4000
        self.random_ = check_random_state(self.random_state)

        self.load_raw_data()
        print("done load raw data")
        self.pre_process()
        print("done pre process")

        # train a classifier to define a logging policy
        self._train_pi_b()

    @property
    def n_actions(self) -> int:
        return int(self.train_label.shape[1])

    def load_raw_data(self) -> None:
        self.train_feature, self.train_label = self._load_raw_data(
            file_path=self.data_path / "train.txt", data_size=self.train_size
        )
        self.test_feature, self.test_label = self._load_raw_data(
            file_path=self.data_path / "test.txt", data_size=self.test_size
        )

    def _load_raw_data(
        self, file_path: PosixPath, data_size: Optional[int] = None
    ) -> tuple[np.ndarray, ...]:
        with open(file_path, "r") as file:
            num_data, num_feature, num_label = file.readline().split()
            num_data = int(num_data) if data_size is None else data_size
            num_feature, num_label = int(num_feature), int(num_label)

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
        self.n_actions = all_label.shape[1]
        
        random_ = check_random_state(self.random_state)
        self.unique_cluster = random_.randint(self.n_clusters, size=self.n_actions)

        # generate reward_noise (depends on each action)
        self.eta = self.random_.uniform(self.max_reward_noise, size=all_label.shape[1])

        self.train_label = sp.csr_matrix(
            all_label[: self.n_train], dtype=np.float32
        ).toarray()
        logits = self.train_label * (1 - self.eta) + (1 - self.train_label) * (
            self.eta - 1
        )
        self.train_expected_rewards = sigmoid(x=logits)

        self.test_label = sp.csr_matrix(
            all_label[self.n_train :], dtype=np.float32
        ).toarray()
        logits = self.test_label * (1 - self.eta) + (1 - self.test_label) * (
            self.eta - 1
        )
        self.test_expected_rewards = sigmoid(x=logits)

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
        self.regressor.fit(self.test_contexts[idx], self.test_expected_rewards[idx])

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
        rewards = self.random_.normal(loc=q_x_a_e_factual, scale=self.reward_std)

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
