from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ope.estimator import InversePropensityScore
from ope.estimator_tuning import BaseOffPolicyEstimatorWithTune
from ope.importance_weight import vanilla_weight
from ope.importance_weight import marginal_weight_over_embedding_spaces
from ope.importance_weight import marginal_weight_over_cluster_spaces
from ope.importance_weight import estimated_marginal_weight


@dataclass
class OffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[InversePropensityScore]
    ope_estimators_tune: Optional[List[BaseOffPolicyEstimatorWithTune]] = None

    def __post_init__(self) -> None:
        self.estimator_names = set(
            [estimator.estimator_name for estimator in self.ope_estimators]
        )
        self.is_model_dependent = False

        if any(name in self.estimator_names for name in ["DM", "DR", "OFFCEM"]):
            self.is_model_dependent = True

    def _create_estimator_inputs(
        self, action_dist: np.ndarray, estimated_rewards: Optional[dict[str, np.ndarray]] = None
    ) -> dict:
        if ("IPS" in self.estimator_names) or ("DR" in self.estimator_names):
            w_x_a = vanilla_weight(data=self.bandit_feedback, action_dist=action_dist)

        if "MIPS (true)" in self.estimator_names or "MDR (true)" in self.estimator_names:
            w_x_e = marginal_weight_over_embedding_spaces(
                data=self.bandit_feedback, 
                action_dist=action_dist
            )

        if ("MIPS" in self.estimator_names) or ("MDR" in self.estimator_names):
            w_x_e_hat = estimated_marginal_weight(
                data=self.bandit_feedback,
                action_dist=action_dist,
                weight_estimator=RandomForestClassifier(n_estimators=10, max_depth=10),
            )
        
        if any("OFFCEM" in name for name in self.estimator_names):
            w_x_c = marginal_weight_over_cluster_spaces(
                data=self.bandit_feedback, 
                action_dist=action_dist
            )
        

        input_data = {}
        for estimator_name in self.estimator_names:
            input_data_ = {}
            # reward
            if estimator_name != "DM":
                input_data_["reward"] = self.bandit_feedback["reward"]

            # weight
            if estimator_name in ["IPS", "DR"]:
                input_data_["weight"] = w_x_a
            elif estimator_name in ["MIPS (true)", "MDR (true)"]:
                input_data_["weight"] = w_x_e
            elif estimator_name in ["MIPS", "MDR"]:
                input_data_["weight"] = w_x_e_hat
            elif "OFFCEM" in estimator_name:
                input_data_["weight"] = w_x_c

            # estimated reward
            if estimator_name == "DR":
                rounds = np.arange(self.bandit_feedback["n_rounds"])
                q_hat = estimated_rewards[estimator_name]
                input_data_["q_hat_factual"] = q_hat[rounds, self.bandit_feedback["action"]]

            if estimator_name in ["DM", "DR"]:
                input_data_["q_hat"] = estimated_rewards[estimator_name]
                input_data_["action_dist"] = action_dist
            
            if "OFFCEM" in estimator_name:
                rounds = np.arange(self.bandit_feedback["n_rounds"])
                f_hat = estimated_rewards[estimator_name]
                input_data_["f_hat_factual"] = f_hat[rounds, self.bandit_feedback["action"]]
                input_data_["f_hat"] = f_hat
                input_data_["action_dist"] = action_dist

            input_data[estimator_name] = input_data_

        return input_data
    

    def estimate_policy_values(
        self, action_dist: np.ndarray, estimated_rewards: Optional[dict[str, np.ndarray]] = None
    ) -> dict:
        if (estimated_rewards is None) and self.is_model_dependent:
            raise ValueError("estimated_rewards must be given")

        input_data = self._create_estimator_inputs(
            action_dist=action_dist, estimated_rewards=estimated_rewards
        )

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(
                **input_data[estimator.estimator_name]
            )
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        if self.ope_estimators_tune is not None:
            for estimator_tune in self.ope_estimators_tune:
                estimated_policy_value = estimator_tune.estimate_policy_value_with_tune(
                    bandit_feedback=self.bandit_feedback, action_dist=action_dist
                )
                estimated_policy_values[
                    estimator_tune.estimator.estimator_name
                ] = estimated_policy_value

        return estimated_policy_values
