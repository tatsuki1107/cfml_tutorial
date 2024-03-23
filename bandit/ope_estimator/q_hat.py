from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LogisticRegressionModel:
    n_action: int
    random_state: int = 12345
    max_iter: int = 1000

    def __post_init__(self) -> None:
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(
        self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:

        rewards_hats = []
        for action in range(self.n_action):
            model_ = self.model
            action_filter = actions == action

            model_.fit(contexts[action_filter], rewards[action_filter])
            rewards_hat = model_.predict_proba(contexts)[:, 1]

            rewards_hats.append(rewards_hat[:, np.newaxis])

        return np.concatenate(rewards_hats, axis=1)
