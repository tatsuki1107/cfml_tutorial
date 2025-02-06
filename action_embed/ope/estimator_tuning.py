from abc import ABCMeta
from abc import abstractmethod

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.base import ClassifierMixin

from action_embed.ope.estimator import MarginalizedIPS


@dataclass
class BaseOffPolicyEstimatorWithTune(metaclass=ABCMeta):
    estimator: MarginalizedIPS
    hyper_param: np.ndarray
    param_name: str

    @abstractmethod
    def estimate_policy_value_with_tune(self):
        raise NotImplementedError


@dataclass
class SLOPE(BaseOffPolicyEstimatorWithTune):
    lower_bound_func: Callable[[np.ndarray, float, bool], np.float64]
    weight_func: Callable[[dict, np.ndarray], np.ndarray]
    tuning_method: str
    weight_estimator: Optional[ClassifierMixin] = None
    min_combination: int = 1
    delta: float = 0.05

    def __post_init__(self) -> None:
        if not self.tuning_method in {
            "exact_scalar",
            "exact_combination",
            "greedy_combination",
        }:
            raise ValueError

    def estimate_policy_value_with_tune(
        self, bandit_feedback: dict, action_dist: np.ndarray
    ) -> np.float64:
        if self.tuning_method == "greedy_combination":
            estimated_policy_value = self._tune_combination_with_greedy_pruning(
                bandit_feedback=bandit_feedback, action_dist=action_dist
            )

        elif self.tuning_method == "exact_combination":
            raise NotImplementedError

        elif self.tuning_method == "exact_scalar":
            raise NotImplementedError

        return estimated_policy_value

    def _tune_combination_with_greedy_pruning(
        self, bandit_feedback: dict, action_dist: np.ndarray
    ):
        theta_list, cnf_list, param_list = [], [], []
        current_param, C = self.hyper_param.copy(), np.sqrt(6) - 1
        bandit_feedback[self.param_name] = current_param

        # init
        kwargs = {
            self.param_name: current_param,
            "weight_estimator": self.weight_estimator,
        }
        importance_weight = self.weight_func(
            data=bandit_feedback, action_dist=action_dist, **kwargs
        )
        theta, cnf = self.estimator.estimate_policy_value_with_dev(
            reward=bandit_feedback["reward"],
            weight=importance_weight,
            lower_bound_func=self.lower_bound_func,
            delta=self.delta,
        )
        theta_list.append(theta), cnf_list.append(cnf)

        current_param_set = set(current_param)
        param_list.append(current_param_set)
        while len(current_param_set) > self.min_combination:
            theta_dict_, cnf_dict_, d_dict_, param_dict_ = {}, {}, {}, {}
            for i, d in enumerate(current_param_set):
                candidate_param = current_param_set.copy()
                candidate_param.remove(d)

                kwargs[self.param_name] = list(candidate_param)
                importance_weight = self.weight_func(
                    data=bandit_feedback, action_dist=action_dist, **kwargs
                )
                theta, cnf = self.estimator.estimate_policy_value_with_dev(
                    reward=bandit_feedback["reward"],
                    weight=importance_weight,
                    lower_bound_func=self.lower_bound_func,
                    delta=self.delta,
                )

                d_dict_[i] = d
                theta_dict_[i] = theta
                cnf_dict_[i] = cnf
                param_dict_[i] = candidate_param.copy()

            idx_list = [
                i
                for i, _ in sorted(cnf_dict_.items(), key=lambda k: k[1], reverse=True)
            ]
            for idx in idx_list:
                excluded_dim, param_i = d_dict_[idx], param_dict_[idx]
                theta_i, cnf_i = theta_dict_[idx], cnf_dict_[idx]
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                self.best_param = param_list[-1]
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                    param_list.append(param_i)
                else:
                    return theta_j[-1]

            current_param_set.remove(excluded_dim)

        return theta_j[-1]

    def _tune_combination_with_exact_pruning(
        self, bandit_feedback: dict, action_dist: np.ndarray
    ):
        pass

    def _tune_scalar_value(self, bandit_feedback: dict, action_dist: np.ndarray):
        pass
