from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ope.estimator import TrajectoryWiseIS as TrajIS


@dataclass
class OffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[TrajIS]

    def __post_init__(self) -> None:
        self.estimator_names = set(
            [estimator.estimator_name for estimator in self.ope_estimators]
        )
        self.is_model_dependent = False

        if "DR" in self.estimator_names:
            self.is_model_dependent = True

    def _create_estimator_inputs(
        self, action_dist: np.ndarray, Q_hat: np.ndarray
    ) -> dict:
        state = self.bandit_feedback["state"]
        action = self.bandit_feedback["action"]
        pscore = self.bandit_feedback["pscore"]

        weight = action_dist[state, action] / pscore

        input_data = {}
        for estimator_name in self.estimator_names:
            input_data_ = {}

            # reward
            input_data_["reward"] = self.bandit_feedback["reward"]
            # weight
            input_data_["weight"] = weight

            if estimator_name == "DR":
                input_data_["Q_hat"] = Q_hat
                input_data_["Q_hat_factual"] = Q_hat[state, action]
                input_data_["action_dist"] = action_dist

            input_data[estimator_name] = input_data_

        return input_data

    def estimate_policy_values(
        self, action_dist: np.ndarray, Q_hat: Optional[np.ndarray] = None
    ) -> dict:
        if action_dist.shape != (
            self.bandit_feedback["n_states"],
            self.bandit_feedback["n_actions"],
        ):
            raise ValueError("action_dist must have the shape (n_states, n_actions).")

        if (Q_hat is None) and self.is_model_dependent:
            raise ValueError("Q_hat must be given.")

        if Q_hat is not None and Q_hat.shape != (
            self.bandit_feedback["n_states"],
            self.bandit_feedback["n_actions"],
        ):
            raise ValueError("Q_hat must have the shape (n_states, n_actions).")

        input_data = self._create_estimator_inputs(action_dist=action_dist, Q_hat=Q_hat)

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(
                **input_data[estimator.estimator_name]
            )
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values
