from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class RegBasedPolicyDataset(Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class GradientBasedPolicyDataset(Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class POTECDataset(Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    f_hat: np.ndarray
    pi_a_x_c: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.f_hat.shape[0]
            == self.pi_a_x_c.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.f_hat[index],
            self.pi_a_x_c[index],
        )

    def __len__(self):
        return self.context.shape[0]
