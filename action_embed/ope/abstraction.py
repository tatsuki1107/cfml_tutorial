from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class ContextActionDataset(Dataset):
    context: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        self.context = torch.tensor(self.context, dtype=torch.float32)
        self.action = torch.tensor(self.action, dtype=torch.float32)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return self.context[idx], self.action[idx]


class ActionEmbeddingModel(nn.Module):
    def __init__(
        self, dim_context: int, n_actions: int, n_cat_dim: int, hidden_size: int
    ) -> None:
        super(ActionEmbeddingModel, self).__init__()
        self.dim_context = dim_context
        self.n_actions = n_actions
        self.n_cat_dim = n_cat_dim
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.dim_context, self.hidden_size)
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding(self.n_actions, self.n_cat_dim)
        self.fc2 = nn.Linear(self.hidden_size + self.n_cat_dim, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = context.size(0)
        x1 = self.fc1(context)
        x1 = self.relu(x1)

        actions = torch.arange(self.n_actions)
        action_embed = self.embedding(actions)

        x1 = x1.unsqueeze(1).repeat(1, self.n_actions, 1)
        action_embed = action_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        x_combined = torch.cat([x1, action_embed], dim=2)
        logits = self.fc2(x_combined).squeeze(2)
        return logits


@dataclass
class AbstractionLearner:
    model: ActionEmbeddingModel
    hidden_size: int
    n_cat_dim: int
    n_cat_per_dim: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    weight_decay: int
    is_discrete: bool = True
    random_state: int = 12345

    def __post_init__(self) -> None:
        # init
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.train_loss, self.val_loss = [], []

    def _create_input_data(
        self, context: np.ndarray, action: np.ndarray, val_size: int
    ) -> tuple[DataLoader]:
        context_train, context_val, action_train, action_val = train_test_split(
            context, action, test_size=val_size, random_state=self.random_state
        )

        train_dataset = ContextActionDataset(context=context_train, action=action_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = ContextActionDataset(context=context_val, action=action_val)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_dataloader, val_dataloader

    def fit(self, context: np.ndarray, action: np.ndarray, val_size: int = 0.2) -> None:
        train_dataloader, val_dataloader = self._create_input_data(
            context=context, action=action, val_size=val_size
        )

        for _ in tqdm(range(self.num_epochs), desc="Training Abstraction Model"):
            self.model.train()
            train_loss_epoch_ = []
            for contexts, actions in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(contexts)

                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()
                train_loss_epoch_.append(loss.item())

            self.train_loss.append(np.mean(train_loss_epoch_))

            self.model.eval()
            val_loss_epoch_ = []
            with torch.no_grad():
                for contexts, actions in val_dataloader:
                    outputs = self.model(contexts)
                    loss = self.criterion(outputs, actions)
                    val_loss_epoch_.append(loss.item())

            self.val_loss.append(np.mean(val_loss_epoch_))

    def obtain_action_embedding(self) -> np.ndarray:
        action_embeddings = self.model.embedding.weight.data.numpy()

        if self.is_discrete:
            discretizer = KBinsDiscretizer(
                n_bins=self.n_cat_per_dim, encode="ordinal", strategy="uniform"
            )
            categorical_action_embedding = []
            for embedding_per_dim in action_embeddings.T:
                categorical_embedding_ = discretizer.fit_transform(
                    embedding_per_dim.reshape(-1, 1)
                ).flatten()
                categorical_action_embedding.append(categorical_embedding_)

            return np.array(categorical_action_embedding).T

        return action_embeddings
