from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from sklearn.utils import check_random_state

from ope.importance_weight import marginal_pscore_over_cluster_spaces
from ope.estimator import InversePropensityScore
from policy.dataset import RegBasedPolicyDataset
from policy.dataset import GradientBasedPolicyDataset
from policy.dataset import POTECDataset
from policy.nn_model import MultiLayerPerceptron as MLP


@dataclass
class BasePolicyLearner(ABC):
    nn_model: MLP
    ope_estimator: InversePropensityScore
    is_conservative: bool = False
    delta: float = 0.05
    batch_size: int = 16
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    regularization_type: str = None
    is_early_stopping: bool = False
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""

        self.random_ = check_random_state(self.random_state)
        self.train_loss = []
        self.test_value = []
        self.test_estimated_value = []

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

    def calc_ground_truth_policy_value(
        self, p_u: np.ndarray, pi_theta: np.ndarray, q_x_a: np.ndarray
    ) -> np.float64:
        return (p_u[:, None] * pi_theta * q_x_a).sum()

    def _judge_early_stopping(self, estimated_policy_value: np.float64) -> bool:
        if self.test_estimated_value:
            last_estimated_value = self.test_estimated_value[-1]
            if estimated_policy_value < last_estimated_value:
                len_ = self.max_iter - len(self.test_estimated_value)
                self.test_estimated_value.extend([last_estimated_value] * len_)

                if self.test_value:
                    last_true_value = self.test_value[-1]
                    self.test_value.extend([last_true_value] * len_)

                return True

        return False

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_train_data_for_opl(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> None:
        raise NotImplementedError


@dataclass
class RegBasedPolicyLearner(BasePolicyLearner):
    """Policy Learner over Action Spaces."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        if self.nn_model.objective != "regression":
            raise ValueError("objective must be regression")

        if self.ope_estimator.decision_making_objective != "action":
            raise ValueError("decision_making_objective must be action")

    def _create_train_data_for_opl(
        self, context: np.ndarray, action: np.ndarray, reward: np.ndarray
    ) -> tuple:
        dataset = RegBasedPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
        )

        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

    def fit(
        self, dataset: dict, dataset_test: dict, true_dist_dict: Optional[dict] = None
    ) -> None:
        context, action, reward = (
            dataset["context"],
            dataset["action"],
            dataset["reward"],
        )
        action_context_ = torch.from_numpy(dataset["action_context_one_hot"]).float()

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=action,
            reward=reward,
        )

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for (
                context_,
                action_,
                reward_,
            ) in training_data_loader:
                optimizer.zero_grad()
                q_hat = self.nn_model(context_, action_context_)
                idx = torch.arange(action_.shape[0], dtype=torch.long)
                loss = ((reward_ - q_hat[idx, action_]) ** 2).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()

            pi_test = self.predict(dataset_test)
            estimated_policy_value = self.estimate_policy_value(
                data=dataset_test, action_dist=pi_test
            )

            if self.is_early_stopping:
                early_stopping_flag = self._judge_early_stopping(estimated_policy_value)
                if early_stopping_flag:
                    break

            self.test_estimated_value.append(estimated_policy_value)

            if true_dist_dict is not None:
                # Note that this is the true policy value, which is not available in practice.
                policy_value = self.calc_ground_truth_policy_value(
                    p_u=true_dist_dict["p_u"],
                    pi_theta=pi_test,
                    q_x_a=true_dist_dict["q_x_a"],
                )
                self.test_value.append(policy_value)

    def predict(self, dataset_test: dict) -> np.ndarray:
        self.nn_model.eval()
        context = torch.from_numpy(dataset_test["x_u"]).float()
        e_a = torch.from_numpy(dataset_test["action_context_one_hot"]).float()
        q_hat = self.nn_model(context, e_a).detach().numpy()
        pi = np.zeros_like(q_hat)
        pi[np.arange(q_hat.shape[0]), np.argmax(q_hat, axis=1)] = 1.0

        return pi

    def estimate_policy_value(self, data: dict, action_dist: np.ndarray) -> np.float64:
        weight = action_dist[data["user_idx"], data["action"]] / data["pscore"]
        reward = data["reward"]

        if self.is_conservative:
            estimated_policy_value = self.ope_estimator.estimate_lower_bound(
                reward=reward, weight=weight, delta=self.delta
            )
        else:
            estimated_policy_value = self.ope_estimator.estimate_policy_value(
                weight=weight, reward=reward
            )

        return estimated_policy_value


@dataclass
class PolicyLearnerOverActionSpaces(BasePolicyLearner):
    """Policy Learner over Action Spaces."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        if self.nn_model.objective != "decision-making":
            raise ValueError("objective must be decision-making")

        if self.ope_estimator.decision_making_objective != "action":
            raise ValueError("decision_making_objective must be action")

    def fit(
        self,
        dataset: dict,
        dataset_test: dict,
        q_hat: Optional[np.ndarray] = None,
        true_dist_dict: Optional[dict] = None,
    ) -> None:
        self.n_actions = dataset["n_actions"]
        context, action, reward = (
            dataset["context"],
            dataset["action"],
            dataset["reward"],
        )
        pscore = dataset["pscore"]
        action_context_ = torch.from_numpy(dataset["action_context_one_hot"]).float()
        if q_hat is None:
            q_hat = np.zeros((reward.shape[0], self.n_actions))

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            q_hat=q_hat,
        )

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for (
                context_,
                action_,
                reward_,
                pscore_,
                q_hat_,
            ) in training_data_loader:
                optimizer.zero_grad()
                pi_theta = self.nn_model(context_, action_context_)
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

            pi_test = self.predict(dataset_test)
            estimated_policy_value = self.estimate_policy_value(
                data=dataset_test, action_dist=pi_test, q_hat=q_hat
            )

            if self.is_early_stopping:
                early_stopping_flag = self._judge_early_stopping(estimated_policy_value)
                if early_stopping_flag:
                    break

            self.test_estimated_value.append(estimated_policy_value)

            if true_dist_dict is not None:
                # Note that this is the true policy value, which is not available in practice.
                policy_value = self.calc_ground_truth_policy_value(
                    p_u=true_dist_dict["p_u"],
                    pi_theta=pi_test,
                    q_x_a=true_dist_dict["q_x_a"],
                )
                self.test_value.append(policy_value)

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

    def estimate_policy_value(
        self, data: dict, action_dist: np.ndarray, q_hat: np.ndarray
    ) -> np.float64:
        weight = action_dist[data["user_idx"], data["action"]] / data["pscore"]
        action_dist_ = action_dist[data["user_idx"]]
        reward = data["reward"]
        q_hat_ = q_hat[data["user_idx"]]
        q_hat_factual = q_hat[data["user_idx"], data["action"]]

        if self.is_conservative:
            estimated_policy_value = self.ope_estimator.estimate_lower_bound(
                reward=reward,
                weight=weight,
                q_hat=q_hat_,
                q_hat_factual=q_hat_factual,
                action_dist=action_dist_,
                delta=self.delta,
            )
        else:
            estimated_policy_value = self.ope_estimator.estimate_policy_value(
                weight=weight,
                reward=reward,
                q_hat=q_hat_,
                q_hat_factual=q_hat_factual,
                action_dist=action_dist_,
            )

        return estimated_policy_value

    def predict(self, dataset_test: dict) -> np.ndarray:
        self.nn_model.eval()
        context = torch.from_numpy(dataset_test["x_u"]).float()
        e_a = torch.from_numpy(dataset_test["action_context_one_hot"]).float()
        return self.nn_model(context, e_a).detach().numpy()

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
        estimated_policy_grad_arr = (
            iw * (reward - q_hat_factual) * log_prob[idx, action]
        )
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, action]

        return estimated_policy_grad_arr


@dataclass
class PolicyLearnerOverActionEmbedSpaces(PolicyLearnerOverActionSpaces):
    """Policy Learner over Action Embedding Spaces."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        if self.nn_model.objective != "decision-making":
            raise ValueError("objective must be decision-making")

        if self.nn_model.name == "two_tower":
            raise ValueError("nn_model must be mlp")

        if self.ope_estimator.decision_making_objective != "action_context":
            raise ValueError("decision_making_objective must be action_context")

        self.n_categories = self.nn_model.dim_output

    def fit(
        self,
        dataset: dict,
        dataset_test: dict,
        q_hat: Optional[np.ndarray] = None,
        true_dist_dict: Optional[dict] = None,
    ) -> None:
        self.n_actions = dataset["n_actions"]

        context, category, reward = (
            dataset["context"],
            dataset["action_context"],
            dataset["reward"],
        )
        pscore_e = dataset["pscore_e"]

        if category.shape[1] != 1:
            raise NotImplementedError

        if q_hat is None:
            q_hat = np.zeros((reward.shape[0], self.n_categories))

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=category,
            action_context=category,
            reward=reward,
            pscore=pscore_e,
            q_hat=q_hat,
        )

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for (
                context_,
                category_,
                _,
                reward_,
                pscore_,
                q_hat_,
            ) in training_data_loader:
                optimizer.zero_grad()
                pi_theta = self.nn_model(context_)
                loss = -self._estimate_policy_gradient(
                    action=category_,
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

            pi_test = self.predict(dataset_test)
            estimated_policy_value = self.estimate_policy_value(
                data=dataset_test, action_dist=pi_test
            )

            if self.is_early_stopping:
                early_stopping_flag = self._judge_early_stopping(estimated_policy_value)
                if early_stopping_flag:
                    break

            self.test_estimated_value.append(estimated_policy_value)

            if true_dist_dict is not None:
                # Note that this is the true policy value, which is not available in practice.
                policy_value = self.calc_ground_truth_policy_value(
                    p_u=true_dist_dict["p_u"],
                    pi_theta=pi_test,
                    q_x_a=true_dist_dict["q_x_a"],
                )
                self.test_value.append(policy_value)

    def predict(self, dataset: dict) -> np.ndarray:
        self.nn_model.eval()
        context = torch.from_numpy(dataset["x_u"]).float()
        p_e_x_pi_theta = self.nn_model(context).detach().numpy()

        n_rounds, e_a = dataset["n_users"], dataset["e_a"]
        overall_policy = np.zeros((n_rounds, self.n_actions))

        best_actions_given_x_c = []
        for e in range(self.n_categories):
            # Assumption of no direct effect.
            best_action_given_x_c = self.random_.choice(
                np.where(e_a == e)[0], size=n_rounds
            )
            best_actions_given_x_c.append(best_action_given_x_c)

        best_actions_given_x_c = np.array(best_actions_given_x_c).T
        overall_policy[
            np.arange(n_rounds)[:, None], best_actions_given_x_c
        ] = p_e_x_pi_theta

        return overall_policy


@dataclass
class PolicyLearnerOverClusterSpaces(BasePolicyLearner):
    """Policy Learner over Action Embedding Spaces."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        if self.nn_model.objective != "decision-making":
            raise ValueError("objective must be decision-making")

        if self.nn_model.name == "two_tower":
            raise ValueError("nn_model must be mlp")

        if self.ope_estimator.decision_making_objective != "cluster":
            raise ValueError("decision_making_objective must be cluster")

    def fit(
        self,
        dataset: dict,
        dataset_test: dict,
        f_hat: Optional[np.ndarray] = None,
        true_dist_dict: Optional[dict] = None,
    ) -> None:
        self.n_actions, self.n_clusters = (
            dataset["n_actions"],
            dataset["n_learned_clusters"],
        )

        context, action, reward = (
            dataset["context"],
            dataset["action"],
            dataset["reward"],
        )
        phi_a = dataset["phi_x_a"][0]

        pscore = marginal_pscore_over_cluster_spaces(
            pi_b=dataset["pi_b"],
            cluster=dataset["cluster"],
            phi_x_a=dataset["phi_x_a"],
        )
        test_pscore = marginal_pscore_over_cluster_spaces(
            pi_b=dataset_test["pi_b"],
            cluster=dataset_test["cluster"],
            phi_x_a=dataset_test["phi_x_a"],
        )

        if f_hat is None:
            f_hat = np.zeros((dataset["n_users"], self.n_actions))

        f_hat_train = f_hat[dataset["user_idx"]]

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=action,
            phi_a=phi_a,
            reward=reward,
            pscore=pscore,
            f_hat=f_hat_train,
        )
        pi_a_x_c = self._calc_2nd_stage_action_dist(f_hat=f_hat, phi_a=phi_a)

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for (
                context_,
                action_,
                reward_,
                pscore_,
                f_hat_,
                pi_a_x_c_,
            ) in training_data_loader:
                optimizer.zero_grad()

                pi_theta = self.nn_model(context_)
                loss = -self._estimate_policy_gradient(
                    action=action_,
                    phi_a=phi_a,
                    reward=reward_,
                    pscore=pscore_,
                    f_hat=f_hat_,
                    pi_a_x_c=pi_a_x_c_,
                    pi_theta=pi_theta,
                ).mean()

                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()

            cluster_dist = self.predict(dataset_test)
            pi_overall = self.predict_overall(dataset_test, pi_a_x_c)
            estimated_policy_value = self.estimate_policy_value(
                dataset_test=dataset_test,
                cluster_dist=cluster_dist,
                f_hat=f_hat,
                pi_overall=pi_overall,
                pscore=test_pscore,
            )

            if self.is_early_stopping:
                early_stopping_flag = self._judge_early_stopping(estimated_policy_value)
                if early_stopping_flag:
                    break
            self.test_estimated_value.append(estimated_policy_value)

            if true_dist_dict is not None:
                # Note that this is the true policy value, which is not available in practice.
                policy_value = self.calc_ground_truth_policy_value(
                    p_u=true_dist_dict["p_u"],
                    pi_theta=pi_overall,
                    q_x_a=true_dist_dict["q_x_a"],
                )
                self.test_value.append(policy_value)

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        phi_a: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        f_hat: np.ndarray,
    ) -> DataLoader:
        pi_a_x_c = self._calc_2nd_stage_action_dist(f_hat=f_hat, phi_a=phi_a)

        dataset = POTECDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(f_hat).float(),
            torch.from_numpy(pi_a_x_c).float(),
        )

        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

    def _calc_2nd_stage_action_dist(self, f_hat: np.ndarray, phi_a) -> np.ndarray:
        action_set = np.arange(self.n_actions)
        datasize = f_hat.shape[0]

        pi_a_x_c = np.zeros((datasize, self.n_actions, self.n_clusters))
        for c in range(self.n_clusters):
            action_mask = phi_a == c
            # assumption of context-free cluster
            a_given_c = f_hat[:, action_mask].argmax(1)
            best_action_given_c = action_set[action_mask][a_given_c]
            pi_a_x_c[np.arange(datasize), best_action_given_c, c] = 1.0

        return pi_a_x_c

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,
        phi_a: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        f_hat: torch.Tensor,
        pi_a_x_c: torch.Tensor,
        pi_theta: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi_theta.detach()
        log_prob = torch.log(pi_theta + self.log_eps)
        idx = torch.arange(action.shape[0], dtype=torch.long)

        f_hat_factual = f_hat[idx, action]
        cluster = phi_a[action]
        iw = current_pi[idx, cluster] / pscore
        estimated_policy_grad_arr = (
            iw * (reward - f_hat_factual) * log_prob[idx, cluster]
        )

        f_hat_c = torch.sum(f_hat[:, :, None] * pi_a_x_c, dim=1)
        estimated_policy_grad_arr += torch.sum(f_hat_c * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, cluster]

        return estimated_policy_grad_arr

    def predict(self, dataset: dict) -> np.ndarray:
        self.nn_model.eval()
        context = torch.from_numpy(dataset["x_u"]).float()
        pi_theta = self.nn_model(context).detach().numpy()
        return pi_theta

    def predict_overall(self, dataset: dict, pi_a_x_c: np.ndarray) -> np.ndarray:
        pi_theta = self.predict(dataset)
        overall_policy = (pi_theta[:, None, :] * pi_a_x_c).sum(2)

        return overall_policy

    def estimate_policy_value(
        self,
        dataset_test: dict,
        cluster_dist: np.ndarray,
        pscore: np.ndarray,
        f_hat: np.ndarray,
        pi_overall: np.ndarray,
    ) -> np.float64:
        user_idx, reward = dataset_test["user_idx"], dataset_test["reward"]
        action, cluster = dataset_test["action"], dataset_test["cluster"]
        weight = cluster_dist[user_idx, cluster] / pscore
        f_hat_ = f_hat[user_idx]
        f_hat_factual = f_hat[user_idx, action]
        pi_overall_ = pi_overall[user_idx]

        if self.is_conservative:
            estimated_policy_value = self.ope_estimator.estimate_lower_bound(
                reward=reward,
                weight=weight,
                f_hat=f_hat_,
                f_hat_factual=f_hat_factual,
                action_dist=pi_overall_,
                delta=self.delta,
            )
        else:
            estimated_policy_value = self.ope_estimator.estimate_policy_value(
                reward=reward,
                weight=weight,
                f_hat=f_hat_,
                f_hat_factual=f_hat_factual,
                action_dist=pi_overall_,
            )

        return estimated_policy_value


@dataclass
class TruePolicyLearner(BasePolicyLearner):
    """Policy Learner over Action Spaces."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        if self.nn_model.objective != "decision-making":
            raise ValueError("objective must be decision-making")

        if self.nn_model.name != "mlp":
            raise ValueError("nn_model must be mlp")

        self.new1_value, self.new2_value = [], []

    def fit(
        self,
        p_u: np.ndarray,
        x_u: np.ndarray,
        q_x_a: np.ndarray,
        squared_q_x_a: np.ndarray,
    ) -> None:
        p_u_ = torch.from_numpy(p_u).float()
        x_u_ = torch.from_numpy(x_u).float()
        q_x_a_ = torch.from_numpy(q_x_a).float()
        squared_q_x_a_ = torch.from_numpy(squared_q_x_a).float()

        # start policy training
        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            self.nn_model.train()
            optimizer.zero_grad()
            pi_theta = self.nn_model(x_u_)
            loss = -self._calc_policy_gradient(
                p_u=p_u_, pi_theta=pi_theta, q_x_a=q_x_a_, squared_q_x_a=squared_q_x_a_
            )

            loss.backward()
            optimizer.step()

            self.train_loss.append(loss.item())
            scheduler.step()

            pi_theta = self.predict(x_u=x_u)

            policy_value = self.calc_ground_truth_policy_value(
                p_u=p_u, pi_theta=pi_theta, q_x_a=q_x_a
            )
            self.test_value.append(policy_value)
            Var_u_a_r = self.calc_var_all(
                p_u=p_u, pi_theta=pi_theta, q_x_a=q_x_a, squared_q_x_a=squared_q_x_a
            )
            self.new1_value.append(Var_u_a_r)
            Var_u_E_a_r = self.calc_var_u(p_u=p_u, pi_theta=pi_theta, q_x_a=q_x_a)
            self.new2_value.append(Var_u_E_a_r)

    def predict(self, x_u: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x_u_ = torch.from_numpy(x_u).float()
        pi_theta = self.nn_model(x_u_).detach().numpy()

        return pi_theta

    def calc_var_all(
        self,
        p_u: np.ndarray,
        pi_theta: np.ndarray,
        q_x_a: np.ndarray,
        squared_q_x_a: np.ndarray,
    ) -> np.float64:
        squared_policy_value = (p_u[:, None] * pi_theta * squared_q_x_a).sum()
        policy_value = (p_u[:, None] * pi_theta * q_x_a).sum()
        variance_all = squared_policy_value - (policy_value**2)
        return variance_all

    def calc_var_u(
        self, p_u: np.ndarray, pi_theta: np.ndarray, q_x_a: np.ndarray
    ) -> np.float64:
        q_u = (pi_theta * q_x_a).sum(1)
        policy_value = (p_u * q_u).sum()
        variance_u = (p_u * (q_u**2)).sum() - (policy_value**2)
        return variance_u

    def _calc_policy_gradient(
        self,
        p_u: torch.Tensor,
        pi_theta: torch.Tensor,
        q_x_a: torch.Tensor,
        squared_q_x_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_pi = pi_theta.detach()
        log_prob = torch.log(pi_theta + self.log_eps)

        policy_gradient = (p_u[:, None] * current_pi * q_x_a * log_prob).sum()

        if self.regularization_type is None:
            return policy_gradient

        # regularization
        # (cfml本 章末問題 5.4)
        if self.regularization_type == "all_variance":
            squared_policy_gradient = (
                p_u[:, None] * current_pi * squared_q_x_a * log_prob
            ).sum()
            policy_value = (p_u[:, None] * current_pi * q_x_a).sum()
            reg_term = squared_policy_gradient - 2 * policy_value * policy_gradient

        elif self.regularization_type == "x_variance":
            q_x = (current_pi * q_x_a).sum(1)
            nabla_q_x_a = (current_pi * q_x_a * log_prob).sum(1)
            first_term = 2 * (p_u * q_x * nabla_q_x_a).sum()

            policy_value = (p_u[:, None] * current_pi * q_x_a).sum()
            second_term = 2 * policy_value * policy_gradient

            reg_term = first_term - second_term

        else:
            raise NotImplementedError

        policy_gradient = policy_gradient - reg_term

        return policy_gradient
    
    def _create_train_data_for_opl(self):
        pass
    
    def estimate_policy_value(self):
        pass
