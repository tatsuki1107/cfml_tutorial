from itertools import permutations
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


@dataclass
class PairWiseDataset(Dataset):
    context: np.ndarray
    action1: np.ndarray
    action2: np.ndarray
    action_emb1: np.ndarray
    action_emb2: np.ndarray
    reward1: np.ndarray
    reward2: np.ndarray

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action1[index],
            self.action2[index],
            self.action_emb1[index],
            self.action_emb2[index],
            self.reward1[index],
            self.reward2[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class PairWiseRegression:
    dim_context: int
    n_actions: int
    n_clusters: int
    hidden_layer_size: tuple = (10, 10, 10)
    activation: str = "elu"
    batch_size: int = 256
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    log_eps: float = 1e-10
    solver: str = "adam"
    max_iter: int = 30
    verbose: bool = False
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        
        self.random_ = check_random_state(self.random_state)
        self.loss_list = []
        
        self.input_size = self.dim_context

        if self.activation == "tanh":
            self.activation_layer = nn.Tanh
        elif self.activation == "relu":
            self.activation_layer = nn.ReLU
        elif self.activation == "elu":
            self.activation_layer = nn.ELU
        
        self.layer_list = []
        for i, h in enumerate(self.hidden_layer_size):
            self.layer_list.append(("l{}".format(i), nn.Linear(self.input_size, h)))
            self.layer_list.append(("a{}".format(i), self.activation_layer()))
            self.input_size = h
        
        self.layer_list.append(("output", nn.Linear(self.input_size, self.n_actions)))
        self.nn_model = nn.Sequential(OrderedDict(self.layer_list))
        

    def _init_scheduler(self) -> tuple[ExponentialLR, optim.Optimizer]:
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")
        
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        
        return scheduler, optimizer


    def fit(self, bandit_data: dict) -> None:
        data_loader = self._make_pairwise_data(bandit_data)
        
        # start pairwise training
        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            losses = []
            self.nn_model.train()
            for x_, a1_, a2_, e1_, e2_, r1_, r2_ in data_loader:
                optimizer.zero_grad()
                loss = self.calc_pairwise_loss(x=x_, a1=a1_, a2=a2_, r1=r1_, r2=r2_)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())
                
            if self.verbose:
                print(_, np.average(losses))
            self.loss_list.append(np.average(losses))
            scheduler.step()
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(context).float()
        h_hat_mat = self.nn_model(x).detach().numpy()
        
        return h_hat_mat
    
    def fit_predict(self, bandit_data: dict) -> np.ndarray:
        self.fit(bandit_data)
        h_hat_mat = self.predict(bandit_data["context"])
        
        return h_hat_mat
    
    def calc_pairwise_loss(self, x: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        h_hat = self.nn_model(x)
        h_hat1, h_hat2 = h_hat[:, a1], h_hat[:, a2]
        loss = ((r1 - r2) - (h_hat1 - h_hat2)) ** 2
        
        return loss.mean()
        

    def _make_pairwise_data(self, bandit_data: dict) -> DataLoader:
        user_idx, actions, rewards = bandit_data["user_idx"], bandit_data["action"], bandit_data["reward"]
        clusters = bandit_data["cluster"]
        fixed_user_contexts = bandit_data["fixed_user_context"]
        fixed_action_contexts = bandit_data["fixed_action_context"]

        reward_dict = defaultdict(dict)
        for user_id, action, reward in zip(user_idx, actions, rewards):
            reward_dict[user_id][action] = reward

        c_a_set_given_x = defaultdict(set)
        for user_id, action, cluster in zip(user_idx, actions, clusters):
            c_a_set_given_x[user_id].add((cluster, action))

        contexts_ = []
        actions1_, actions2_ = [], []
        action_contexts1_, action_contexts2_ = [], []
        rewards1_, rewards2_ = [], []

        for u in np.unique(user_idx):

            a_set_given_c = defaultdict(set)
            for cluster, action in c_a_set_given_x[u]:
                a_set_given_c[cluster].add(action)

            for c, obs_actions_in_c in a_set_given_c.items():
                for (a1, a2) in permutations(obs_actions_in_c, 2):
                    r1, r2 = reward_dict[u][a1], reward_dict[u][a1]
                    contexts_.append(fixed_user_contexts[u])
                    actions1_.append(a1), actions2_.append(a2)
                    action_contexts1_.append(fixed_action_contexts[a1])
                    action_contexts2_.append(fixed_action_contexts[a2])
                    rewards1_.append(r1), rewards2_.append(r2)


        pairwise_dataset = PairWiseDataset(
            torch.from_numpy(np.array(contexts_)).float(),
            torch.from_numpy(np.array(actions1_)).long(),
            torch.from_numpy(np.array(actions2_)).long(),
            torch.from_numpy(np.array(action_contexts1_)).float(),
            torch.from_numpy(np.array(action_contexts2_)).float(),
            torch.from_numpy(np.array(rewards1_)).float(),
            torch.from_numpy(np.array(rewards2_)).float(),
        )

        data_loader = DataLoader(
            pairwise_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        return data_loader
