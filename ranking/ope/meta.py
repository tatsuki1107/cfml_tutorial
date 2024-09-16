from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ope.estimator import InversePropensityScore
from ope.impotance_weight import vanilla_weight
from ope.impotance_weight import independent_weight
from ope.impotance_weight import cascade_weight
from ope.impotance_weight import adaptive_weight


@dataclass
class RankingOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[InversePropensityScore]
    ope_estimators_tune: Optional[List[InversePropensityScore]] = None
    alpha: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.estimator_names = set(
            [estimator.estimator_name for estimator in self.ope_estimators]
        )

        if self.alpha is None:
            self.alpha = np.ones(self.bandit_feedback["len_list"], dtype=float)

    def _create_estimator_inputs(self, action_dist: np.ndarray) -> dict:
        if ("RIPS" in self.estimator_names) or ("Cascade-DR" in self.estimator_names):
            w_x_a_1_k = cascade_weight(
                data=self.bandit_feedback, action_dist=action_dist
            )

        input_data = {}
        for estimator_name in self.estimator_names:
            input_data_ = {}
            # reward
            input_data_["reward"] = self.bandit_feedback["reward"]

            # weight
            if estimator_name == "SIPS":
                weight = vanilla_weight(
                    data=self.bandit_feedback, action_dist=action_dist
                )
            elif estimator_name == "IIPS":
                weight = independent_weight(
                    data=self.bandit_feedback, action_dist=action_dist
                )
            elif estimator_name in {"RIPS", "Cascade-DR"}:
                weight = w_x_a_1_k
            elif estimator_name == "AIPS (true)":
                weight = adaptive_weight(
                    data=self.bandit_feedback, action_dist=action_dist
                )

            input_data_["weight"] = weight
            input_data_["alpha"] = self.alpha
            input_data[estimator_name] = input_data_

        return input_data

    def estimate_policy_values(self, action_dist: np.ndarray) -> dict:
        input_data = self._create_estimator_inputs(action_dist=action_dist)

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(
                **input_data[estimator.estimator_name]
            )
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        if self.ope_estimators_tune:
            raise NotImplementedError

        return estimated_policy_values
