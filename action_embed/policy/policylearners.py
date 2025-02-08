from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import check_random_state


@dataclass
class RegBasedPolicyDataset(Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
        )

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
class BaseGradientBasedPolicyLearner(ABC):
    dim_context: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        
        self.random_ = check_random_state(self.random_state)
        self.train_loss = []
        self.train_value = []
        self.test_value = []
        
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

    @abstractmethod
    def fit(self) -> None:
        pass

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi_theta: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi_theta.detach()
        log_prob = torch.log(pi_theta + self.log_eps)
        idx = torch.arange(action.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, action]
        iw = current_pi[idx, action] / pscore
        estimated_policy_grad_arr = iw * (reward - q_hat_factual) * log_prob[idx, action]
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, action]

        return estimated_policy_grad_arr
    
    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass


class RegBasedPolicyLearner(BaseGradientBasedPolicyLearner):
    """Policy Learner over Action Spaces."""
    num_actions: int
    
    def __init__(self, num_actions: int, **kwargs) -> None:
        """Initialize class."""
        self.num_actions = num_actions
        super().__init__(**kwargs)
    
    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        self.layer_list.append(("output", nn.Linear(self.input_size, self.num_actions)))
        self.nn_model = nn.Sequential(OrderedDict(self.layer_list))
    
    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
    ) -> tuple:
        dataset = RegBasedPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
        )

        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader
    
    def fit(self, dataset: dict, dataset_test: dict) -> None:
        context, action, reward = dataset["context"], dataset["action"], dataset["reward"]

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=action,
            reward=reward,
        )

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        q_x_a_train, q_x_a_test = dataset["expected_reward"], dataset_test["expected_reward"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for context_, action_, reward_, in training_data_loader:
                optimizer.zero_grad()
                q_hat = self.nn_model(context_)
                idx = torch.arange(action_.shape[0], dtype=torch.long)
                loss = ((reward_ - q_hat[idx, action_]) ** 2).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
    
    def predict(self, dataset_test: dict) -> np.ndarray:

        self.nn_model.eval()
        context = torch.from_numpy(dataset_test["context"]).float()
        q_hat = self.nn_model(context).detach().numpy()
        pi = np.zeros_like(q_hat)
        pi[np.arange(q_hat.shape[0]), np.argmax(q_hat, axis=1)] = 1.0
        
        return pi


class PolicyLearnerOverActionSpaces(BaseGradientBasedPolicyLearner):
    """Policy Learner over Action Spaces."""
    num_actions: int
    
    def __init__(self, num_actions: int, **kwargs) -> None:
        """Initialize class."""
        self.num_actions = num_actions
        super().__init__(**kwargs)
    
    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        self.layer_list.append(("output", nn.Linear(self.input_size, self.num_actions)))
        self.layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(self.layer_list))
    
    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        context, action, reward = dataset["context"], dataset["action"], dataset["reward"]
        pscore = dataset["pscore"]
        if q_hat is None:
            q_hat = np.zeros((reward.shape[0], self.num_actions))

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            q_hat=q_hat,
        )

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        q_x_a_train, q_x_a_test = dataset["expected_reward"], dataset_test["expected_reward"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for context_, action_, reward_, pscore_, q_hat_, in training_data_loader:
                optimizer.zero_grad()
                pi_theta = self.nn_model(context_)
                loss = -self._estimate_policy_gradient(
                    action=action_,
                    reward=reward_,
                    pscore=pscore_,
                    q_hat=q_hat_,
                    pi_theta=pi_theta,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
    
    def predict(self, dataset_test: dict) -> np.ndarray:

        self.nn_model.eval()
        context = torch.from_numpy(dataset_test["context"]).float()
        return self.nn_model(context).detach().numpy()


@dataclass(init=False)
class PolicyLearnerOverActionEmbedSpaces(BaseGradientBasedPolicyLearner):
    """Policy Learner over Action Embedding Spaces."""
    num_actions: int
    num_category: int
    dim_action_context: int
    is_discrete: bool = True
    is_probabilistic: bool = False
    
    def __init__(
        self, 
        num_actions: int, 
        num_category: int, 
        dim_action_context: int, 
        **kwargs
    ) -> None:
        """Initialize class."""
        self.num_actions = num_actions
        self.num_category = num_category
        self.dim_action_context = dim_action_context
        super().__init__(**kwargs)
    
    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        
        if self.is_discrete and (not self.is_probabilistic):

            self.layer_list.append(("output", nn.Linear(self.input_size, self.num_category)))
            self.layer_list.append(("softmax", nn.Softmax(dim=1)))

            self.nn_model = nn.Sequential(OrderedDict(self.layer_list))
        
        else:
            raise NotImplementedError
    
    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        context, category, reward = dataset["context"], dataset["action_context"], dataset["reward"]
        pscore_e = dataset["pscore_e"]
        
        if category.shape[1] != 1:
            raise NotImplementedError
        
        category = category.reshape(-1)
        
        if q_hat is None:
            q_hat = np.zeros((reward.shape[0], self.num_category))

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=category,
            reward=reward,
            pscore=pscore_e,
            q_hat=q_hat,
        )

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        q_x_a_train, q_x_a_test = dataset["expected_reward"], dataset_test["expected_reward"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for context_, category_, reward_, pscore_e_, q_hat_, in training_data_loader:
                optimizer.zero_grad()
                pi_theta = self.nn_model(context_)
                loss = -self._estimate_policy_gradient(
                    action=category_,
                    reward=reward_,
                    pscore=pscore_e_,
                    q_hat=q_hat_,
                    pi_theta=pi_theta,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
    
    def predict(self, dataset: dict) -> np.ndarray:

        self.nn_model.eval()
        context = torch.from_numpy(dataset["context"]).float()
        p_e_x_pi_theta = self.nn_model(context).detach().numpy()
        
        n_rounds, e_a = dataset["n_rounds"], dataset["e_a"]
        overall_policy = np.zeros((n_rounds, self.num_actions))
        
        best_actions_given_x_c = []
        for e in range(self.num_category):
            # Assumption of no direct effect.
            best_action_given_x_c = self.random_.choice(np.where(e_a == e)[0], size=n_rounds)
            best_actions_given_x_c.append(best_action_given_x_c)
        
        best_actions_given_x_c = np.array(best_actions_given_x_c).T
        overall_policy[np.arange(n_rounds)[:, None], best_actions_given_x_c] = p_e_x_pi_theta
        
        return overall_policy
