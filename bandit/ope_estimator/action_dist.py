from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, FloatTensor


@dataclass(unsafe_hash=True)
class NNMultiClassifer(nn.Module):

    input_size: int
    hidden_layer_sizes: Tuple[int, ...]
    num_class: int
    seed: int

    def __post_init__(self) -> None:
        super().__init__()
        torch.manual_seed(self.seed)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(self.input_size, self.hidden_layer_sizes[0])
        )
        for hin, hout in zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:]):
            self.hidden_layers.append(nn.Linear(hin, hout))
        self.output = nn.Linear(self.hidden_layer_sizes[-1], self.num_class)

    def forward(self, x: FloatTensor) -> FloatTensor:
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))

        y_hat = F.softmax(self.output(h), dim=1)
        return y_hat


@dataclass
class NNTrainer:
    input_size: int
    hidden_layer_sizes: Tuple[int, ...]
    num_class: int
    seed: int
    lr: float
    reg: float
    num_epoch: int
    train_ratio: float = 0.8

    def __post_init__(self) -> None:
        self.model = NNMultiClassifer(
            input_size=self.input_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_class=self.num_class,
            seed=self.seed,
        )
        self.train_loss = []
        self.val_loss = []

    def fit(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        # all data
        _contexts = torch.tensor(contexts, dtype=torch.float)
        _actions = torch.tensor(actions, dtype=torch.long)
        _actions = F.one_hot(_actions, num_classes=self.num_class)

        (train_contexts, train_actions, val_contexts, val_actions) = (
            self._split_dataset(_contexts, _actions)
        )

        optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=self.reg
        )

        for _ in range(self.num_epoch):
            self.model.train()
            train_pi_b_hats = self.predict_proba(train_contexts)
            train_logloss = self._calculate_logloss(train_pi_b_hats, train_actions)
            self.train_loss.append(train_logloss.item())

            optimizer.zero_grad()
            train_logloss.backward()
            optimizer.step()

            self.model.eval()
            val_pi_b_hats = self.predict_proba(val_contexts)
            val_logloss = self._calculate_logloss(val_pi_b_hats, val_actions)
            self.val_loss.append(val_logloss.item())

        pi_b_hats = self.predict_proba(_contexts)
        pi_b_hats = pi_b_hats.detach().numpy()

        indices = np.arange(len(actions))
        return pi_b_hats[indices, actions]

    def predict_proba(self, contexts: torch.Tensor) -> torch.Tensor:
        pi_b = self.model(contexts)
        return pi_b

    def _calculate_logloss(
        self, pi_b: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        logloss = -torch.sum(actions * torch.log(pi_b)) / len(actions)
        return logloss

    def _split_dataset(
        self, contexts: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        suffle_idx = torch.randperm(len(contexts))
        train_size = int(len(contexts) * self.train_ratio)
        train_idx = suffle_idx[:train_size]
        val_idx = suffle_idx[train_size:]

        # train data
        train_contexts = contexts[train_idx]
        train_actions = actions[train_idx]
        # val data
        val_contexts = contexts[val_idx]
        val_actions = actions[val_idx]

        return train_contexts, train_actions, val_contexts, val_actions


@dataclass
class GBCModel:
    n_estimators: int
    max_depth: int
    lr: float
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.model = GBC(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            random_state=self.random_state,
        )

    def fit(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:

        self.model.fit(contexts, actions)
        pi_b_hats = self.model.predict_proba(contexts)

        indices = np.arange(len(actions))
        return pi_b_hats[indices, actions]
