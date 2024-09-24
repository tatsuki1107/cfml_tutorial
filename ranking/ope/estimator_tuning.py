from abc import ABCMeta
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional


import numpy as np
from sklearn.utils import check_random_state

from dataset import SyntheticRankingBanditDataset
from ope.estimator import InversePropensityScore
from ope.importance_weight import vanilla_weight
from ope.importance_weight import adaptive_weight
from utils.policy import gen_eps_greedy
from utils.user_behavior import parse_behavior


@dataclass
class BaseOfflineEstimatorWithSurrogateMSE(metaclass=ABCMeta):
    estimator: InversePropensityScore
    param_name: str
    bias_estimation_method: str

    @abstractmethod
    def estimate_policy_value_with_tune(self):
        raise NotImplementedError


@dataclass
class Node:
    node_id: int
    user_behavior_id: int
    depth: int
    sample_id: np.ndarray
    bootstrap_sample_ids: dict[int, np.ndarray]


@dataclass
class UserBehaviorTree(BaseOfflineEstimatorWithSurrogateMSE):
    dataset: SyntheticRankingBanditDataset
    weight_func: adaptive_weight
    candidate_models: dict[str, vanilla_weight]
    eps: float
    val_size: int
    len_list: int
    n_partition: int = 5
    min_samples_leaf: int = 10
    n_bootstrap: int = 10
    alpha: Optional[np.ndarray] = None
    max_depth: Optional[int] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.estimator.estimator_name != "AIPS - tree":
            raise ValueError

        self.decision_boundary = dict()

        self.n_candidate_models = len(self.candidate_models)
        self.id2behavior = {
            i: c for i, (c, _) in enumerate(self.candidate_models.items())
        }
        self.candidate_models = {
            i: func for i, (_, func) in enumerate(self.candidate_models.items())
        }

        if self.max_depth is None:
            self.max_depth = np.infty

        if self.alpha is None:
            self.alpha = np.ones(self.len_list)

        if self.bias_estimation_method == "experimental_on_policy":
            on_data = self.dataset.obtain_batch_bandit_feedback(
                n_rounds=self.val_size, is_online=True
            )
            self.approximate_policy_value = (
                (self.alpha * on_data["reward"]).sum(1).mean()
            )
        else:
            raise NotImplementedError

        self.random_ = check_random_state(self.random_state)

    def estimate_policy_value_with_tune(
        self, data: dict, action_dist: np.ndarray
    ) -> np.float64:
        estimated_user_behavior = self.fit_predict(data=data, action_dist=action_dist)
        kwargs_ = {self.param_name: estimated_user_behavior}
        importance_weight = self.weight_func(
            data=data, action_dist=action_dist, **kwargs_
        )

        estimated_value = self.estimator.estimate_policy_value(
            weight=importance_weight, alpha=self.alpha, reward=data["reward"]
        )

        return estimated_value

    def fit(self, data: dict, action_dist: np.ndarray) -> None:
        self._create_input_dataset(data=data, action_dist=action_dist)

        # init
        node_queue = deque()
        base_user_behavior_id = self._select_base_user_behavior()
        node_id, depth = 0, 0
        sample_id = np.arange(self.train_size)
        bootstrap_sample_ids = {
            i: np.arange(self.train_size) for i in range(self.n_bootstrap)
        }

        initial_node = Node(
            node_id=node_id,
            sample_id=sample_id,
            user_behavior_id=base_user_behavior_id,
            depth=depth,
            bootstrap_sample_ids=bootstrap_sample_ids,
        )
        node_queue.append(initial_node)
        self._update_decision_boundary(
            node_id=node_id, user_behavior_id=base_user_behavior_id
        )
        self._update_global_pscore(node=initial_node)

        while len(node_queue):
            parent_node: Node = node_queue.pop()
            split_outcomes = self._search_split_boundary(parent_node)

            if split_outcomes["split_exist"]:
                # initialize child node
                left_node_id, right_node_id = node_id + 1, node_id + 2
                child_node_depth = parent_node.depth + 1
                left_node = Node(
                    node_id=left_node_id,
                    sample_id=split_outcomes["left_sample_id"],
                    user_behavior_id=split_outcomes["left_user_behavior_id"],
                    depth=child_node_depth,
                    bootstrap_sample_ids=split_outcomes["left_bootstrap_sample_ids"],
                )
                right_node = Node(
                    node_id=right_node_id,
                    sample_id=split_outcomes["right_sample_id"],
                    user_behavior_id=split_outcomes["right_user_behavior_id"],
                    depth=child_node_depth,
                    bootstrap_sample_ids=split_outcomes["right_bootstrap_sample_ids"],
                )
                self._update_decision_boundary(
                    node_id=left_node_id,
                    user_behavior_id=split_outcomes["left_user_behavior_id"],
                )
                self._update_decision_boundary(
                    node_id=right_node_id,
                    user_behavior_id=split_outcomes["right_user_behavior_id"],
                )
                self._update_global_pscore(left_node)
                self._update_global_pscore(right_node)

                # increment
                node_queue.append(left_node)
                node_queue.append(right_node)
                node_id += 2

    def fit_predict(self, data: dict, action_dist: np.ndarray) -> np.ndarray:
        self.fit(data=data, action_dist=action_dist)

        estimated_user_behavior = np.array(
            [self.id2behavior[i] for i in self.train_reward_structure]
        )
        return estimated_user_behavior

    def predict(self, context: np.ndarray):
        raise NotImplementedError

    def _search_split_boundary(self, parent_node: Node) -> dict:
        parent_node_id = parent_node.node_id
        parent_sample_id = parent_node.sample_id
        parent_bootstrap_sample_ids = parent_node.bootstrap_sample_ids
        parent_user_behavior_id = parent_node.user_behavior_id
        parent_depth = parent_node.depth
        parent_context = self.train_context[parent_sample_id]

        n_parent_samples = len(parent_sample_id)
        if n_parent_samples < 2 * self.min_samples_leaf:
            return {"split_exist": False}
        if parent_depth == self.max_depth:
            return {"split_exist": False}

        # init
        best_outcome = dict()
        best_outcome["split_exist"] = False
        best_mse = self._calc_mse_global()
        best_outcome["mse"] = best_mse

        split_feature_dims = self.random_.choice(
            self.train_context.shape[1], size=self.n_partition, replace=True
        )
        min_left_proportion = self.min_samples_leaf / n_parent_samples
        max_left_proportion = 1 - min_left_proportion
        split_left_proportions = self.random_.uniform(
            min_left_proportion, max_left_proportion, size=self.n_partition
        )
        feature_dim_sort_idx = np.argsort(split_feature_dims)
        split_feature_dims = split_feature_dims[feature_dim_sort_idx]
        split_left_proportions = split_left_proportions[feature_dim_sort_idx]

        for i in range(self.n_partition):
            feature_dim = split_feature_dims[i]
            if (i == 0) or (feature_dim != split_feature_dims[i - 1]):
                sorted_sample_id = parent_sample_id[
                    np.argsort(parent_context[:, feature_dim])
                ]

            split_id = int(split_left_proportions[i] * n_parent_samples)
            left_sample_id = sorted_sample_id[:split_id]
            right_sample_id = sorted_sample_id[split_id:]

            split_feature_value = (
                self.train_context[left_sample_id[-1], feature_dim]
                + self.train_context[right_sample_id[0], feature_dim]
            ) / 2

            (
                left_user_behavior_id,
                right_user_behavior_id,
                left_bootstrap_sample_ids,
                right_bootstrap_sample_ids,
                split_mse,
            ) = self._find_best_user_behavior(
                parent_sample_ids=parent_bootstrap_sample_ids,
                split_feature_dim=feature_dim,
                split_feature_value=split_feature_value,
            )
            if split_mse <= best_outcome["mse"]:
                best_outcome["split_exist"] = True
                best_outcome["mse"] = split_mse
                best_outcome["split_feature_dim"] = feature_dim
                best_outcome["split_feature_value"] = split_feature_value
                best_outcome["left_sample_id"] = left_sample_id
                best_outcome["right_sample_id"] = right_sample_id
                best_outcome["left_bootstrap_sample_ids"] = left_bootstrap_sample_ids
                best_outcome["right_bootstrap_sample_ids"] = right_bootstrap_sample_ids
                best_outcome["left_user_behavior_id"] = left_user_behavior_id
                best_outcome["right_user_behavior_id"] = right_user_behavior_id

        if best_outcome["split_exist"]:
            self._update_decision_boundary(
                node_id=parent_node_id,
                user_behavior_id=parent_user_behavior_id,
                split_exist=True,
                feature_dim=best_outcome["split_feature_dim"],
                feature_value=best_outcome["split_feature_value"],
            )

        return best_outcome

    def _find_best_user_behavior(
        self,
        parent_sample_ids: dict[int, np.ndarray],
        split_feature_dim: int,
        split_feature_value: float,
    ) -> tuple:
        estimated_values = np.zeros(
            (self.n_candidate_models, self.n_candidate_models, self.n_bootstrap)
        )
        left_sample_ids, right_sample_ids = {}, {}
        for i, bootstrap_data_ in enumerate(self.bootstrap_dataset):
            context, reward = bootstrap_data_["context"], bootstrap_data_["reward"]
            importance_weight = bootstrap_data_["importance_weight_dict"]
            weighted_reward_ = bootstrap_data_["weighted_reward"].copy()

            sample_id = parent_sample_ids[i].copy()
            context = context[sample_id]

            left_context_mask = np.where(
                context[:, split_feature_dim] < split_feature_value
            )
            right_context_mask = np.where(
                context[:, split_feature_dim] >= split_feature_value
            )
            left_sample_id_, right_sample_id_ = (
                sample_id[left_context_mask],
                sample_id[right_context_mask],
            )
            left_sample_ids[i] = left_sample_id_
            right_sample_ids[i] = right_sample_id_

            for left_user_behavior_id in range(self.n_candidate_models):
                weighted_reward_[
                    left_sample_id_
                ] = self.estimator._estimate_round_rewards(
                    weight=importance_weight[left_user_behavior_id][left_sample_id_],
                    alpha=self.alpha,
                    reward=reward[left_sample_id_],
                )

                for right_user_behavior_id in range(self.n_candidate_models):
                    weighted_reward_[
                        right_sample_id_
                    ] = self.estimator._estimate_round_rewards(
                        weight=importance_weight[right_user_behavior_id][
                            right_sample_id_
                        ],
                        alpha=self.alpha,
                        reward=reward[right_sample_id_],
                    )
                    estimated_values[
                        left_user_behavior_id, right_user_behavior_id, i
                    ] = weighted_reward_.mean()

        surrogate_mse = self._compute_surrogate_mse(estimated_values, axis=2)
        best_left_user_behavior, best_right_user_behavior = np.unravel_index(
            surrogate_mse.argmin(), surrogate_mse.shape
        )
        split_outcome = (
            best_left_user_behavior,
            best_right_user_behavior,
            left_sample_ids,
            right_sample_ids,
            surrogate_mse.min(),
        )
        return split_outcome

    def _calc_mse_global(self) -> None:
        estimated_values = []
        for bootstrap_data_ in self.bootstrap_dataset:
            estimated_values.append(bootstrap_data_["weighted_reward"].mean())

        estimated_values = np.array(estimated_values)
        surrogate_mse = self._compute_surrogate_mse(estimated_values, axis=0)
        return surrogate_mse

    def _update_global_pscore(self, node: Node) -> None:
        user_behavior_id = node.user_behavior_id
        sample_id = node.sample_id
        bootstrap_sample_ids = node.bootstrap_sample_ids

        self.train_reward_structure[sample_id] = user_behavior_id
        self.train_weighted_reward[sample_id] = self.estimator._estimate_round_rewards(
            weight=self.train_importance_weight_dict[user_behavior_id][sample_id],
            alpha=self.alpha,
            reward=self.train_reward[sample_id],
        )

        for i, bootstrap_data_ in enumerate(self.bootstrap_dataset):
            sample_id = bootstrap_sample_ids[i]
            iw_dict, reward = (
                bootstrap_data_["importance_weight_dict"],
                bootstrap_data_["reward"],
            )

            bootstrap_data_["reward_structure"][sample_id] = user_behavior_id
            bootstrap_data_["weighted_reward"][
                sample_id
            ] = self.estimator._estimate_round_rewards(
                weight=iw_dict[user_behavior_id][sample_id],
                alpha=self.alpha,
                reward=reward[sample_id],
            )

    def _update_decision_boundary(
        self,
        node_id: int,
        user_behavior_id: int,
        split_exist: Optional[bool] = False,
        feature_dim: Optional[int] = None,
        feature_value: Optional[float] = None,
    ) -> None:
        self.decision_boundary[node_id] = {
            "parent_user_behavior_id": user_behavior_id,
            "split_exist": split_exist,
            "feature_dim": feature_dim,
            "feature_value": feature_value,
        }

    def _select_base_user_behavior(self) -> np.int64:
        estimated_values = np.zeros((self.n_candidate_models, self.n_bootstrap))
        for i, bootstrap_data_ in enumerate(self.bootstrap_dataset):
            reward = bootstrap_data_["reward"]
            importance_weight_dict = bootstrap_data_["importance_weight_dict"]

            for user_behavior_id, iw in importance_weight_dict.items():
                estimated_value = self.estimator.estimate_policy_value(
                    weight=iw, alpha=self.alpha, reward=reward
                )
                estimated_values[user_behavior_id, i] = estimated_value

        surrogate_mse = self._compute_surrogate_mse(
            estimated_values=estimated_values, axis=1
        )
        best_user_behavior_id = surrogate_mse.argmin()
        return best_user_behavior_id

    def _compute_surrogate_mse(
        self, estimated_values: np.ndarray, axis: int
    ) -> np.ndarray:
        bias = estimated_values.mean(axis) - self.approximate_policy_value
        variance = estimated_values.var(axis)
        surrogate_mse = (bias**2) + variance
        return surrogate_mse

    def _create_input_dataset(self, data: dict, action_dist: np.ndarray) -> None:
        self.train_size = len(data["context"])
        self._create_bootstrap_dataset()

        self.train_importance_weight_dict = self._compute_initialized_importance_weight(
            data=data, action_dist=action_dist
        )
        self.train_context = data["context"]
        self.train_reward = data["reward"]
        self.train_reward_structure = np.zeros(self.train_size, dtype=int)
        self.train_weighted_reward = np.zeros(self.train_size)

    def _create_bootstrap_dataset(self) -> None:
        self.bootstrap_dataset = []
        for _ in range(self.n_bootstrap):
            bootstrap_data_ = self.dataset.obtain_batch_bandit_feedback(
                n_rounds=self.train_size
            )
            policy_logit_ = bootstrap_data_["evaluation_policy_logit"]
            action_dist_ = gen_eps_greedy(expected_reward=policy_logit_, eps=self.eps)

            importance_weight_dict_ = self._compute_initialized_importance_weight(
                data=bootstrap_data_, action_dist=action_dist_
            )

            self.bootstrap_dataset.append(
                dict(
                    context=bootstrap_data_["context"],
                    reward=bootstrap_data_["reward"],
                    importance_weight_dict=importance_weight_dict_,
                    reward_structure=np.zeros(self.train_size, dtype=int),
                    weighted_reward=np.zeros(self.train_size),
                )
            )

    def _compute_initialized_importance_weight(
        self, data: dict, action_dist: np.ndarray
    ) -> dict:
        importance_weight_dict = dict()
        for user_behavior_id, weight_func in self.candidate_models.items():
            input_data = dict(data=data, action_dist=action_dist)
            parsed_behavior = parse_behavior(self.id2behavior[user_behavior_id])
            if parsed_behavior:
                _, k = parsed_behavior
                input_data["k"] = k

            importance_weight_dict[user_behavior_id] = weight_func(**input_data)

        return importance_weight_dict
