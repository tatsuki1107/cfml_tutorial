from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR


@dataclass
class LogisticRegressionModel:
    n_action: int
    random_state: int = 12345
    max_iter: int = 1000
    solver: str = "liblinear"

    def __post_init__(self) -> None:
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
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


@dataclass
class GBRModel:
    n_action: int
    n_estimators: int
    max_depth: int
    lr: float
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.model = GBR(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            random_state=self.random_state,
        )

        self.action_contexts = np.eye(self.n_action)

    def _pre_process_for_reg_model(
        self, contexts: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        return np.c_[contexts, self.action_contexts[actions]]

    def fit(
        self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> None:
        contexts_ = self._pre_process_for_reg_model(contexts, actions)
        self.model.fit(contexts_, rewards)

    def predict(self, contexts: np.ndarray) -> np.ndarray:
        actions_ = np.arange(self.n_action)
        rewards_hats = []
        for context in contexts:
            context_ = self._pre_process_for_reg_model(
                contexts=np.tile(context, (self.n_action, 1)), actions=actions_
            )
            rewards_hat = self.model.predict(context_)
            rewards_hats.append(rewards_hat)

        return np.array(rewards_hats)

    def fit_predict(
        self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:

        self.fit(contexts=contexts, actions=actions, rewards=rewards)
        rewards_hats = self.predict(contexts=contexts)

        return rewards_hats
