from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from obp.utils import softmax, sample_action_fast
from obp.dataset import linear_reward_function
from obp.dataset import OpenBanditDataset
from obp.types import BanditFeedback


class BaseBanditDataset(metaclass=ABCMeta):
    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        raise NotImplementedError


@dataclass
class SyntheticBanditDatasetWithActionEmbeds(BaseBanditDataset):

    n_actions: int
    dim_context: int
    n_cat_dim: int
    n_cat_per_dim: int
    latent_param_mat_dim: int
    n_unobserved_cat_dim: int = 0
    p_e_a_param_std: float = 1.0
    reward_noise: float = 1.0
    beta: float = -3.0
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self._define_action_embed()
    
    def _define_action_embed(self) -> None:
        
        # p(e_d|x,a)
        self.p_e_d_a = softmax(
            self.random_.normal(
                scale=self.p_e_a_param_std, 
                size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim)
            )
        )
        
        self.latent_cat_param = self.random_.normal(
            size=(self.n_cat_dim, self.n_cat_per_dim, self.latent_param_mat_dim)
        )
        
        self.cat_dim_importance = self.random_.dirichlet(
            alpha=self.random_.uniform(size=self.n_cat_dim),
            size=1,
        )

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:

        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))
        
        # r
        q_x_a, q_x_e = [], []
        for d in range(self.n_cat_dim):
            q_x_e_d = linear_reward_function(
                context=context,
                action_context=self.latent_cat_param[d],
                random_state=self.random_state + d,
            )
            q_x_a_d = q_x_e_d @ self.p_e_d_a[:, :, d].T

            q_x_a.append(q_x_a_d)
            q_x_e.append(q_x_e_d)

        q_x_a = np.array(q_x_a).transpose(1, 2, 0) # q_x_a.shape: (n_rounds, n_actions, n_cat_dim) 
        q_x_e = np.array(q_x_e).transpose(1, 0, 2) # q_x_e.shape: (n_rounds, n_cat_dim, n_cat_per_dim)
        
        cat_dim_importance_ = self.cat_dim_importance.reshape((1, 1, self.n_cat_dim))
        # shape: (n_rounds, n_actions)
        q_x_a = (q_x_a * cat_dim_importance_).sum(2)
        
        
        pi_b = softmax(self.beta * q_x_a)
        # a ~ \pi_b(\cdot|x)
        action = sample_action_fast(pi_b)
        
        action_context = []
        for d in range(self.n_cat_dim):
            # e_d ~ p(e_d|x,a), e ~ \prod_{d} p(e_d|x,a)
            action_context_d = sample_action_fast(self.p_e_d_a[action, :, d])
            action_context.append(action_context_d)

        action_context = np.array(action_context).T
        
        cat_dim_importance_ = self.cat_dim_importance.reshape((1, self.n_cat_dim, 1))
        expected_reward_factual = (cat_dim_importance_ * q_x_e)[
            np.arange(n_rounds)[:,None], np.arange(self.n_cat_dim), action_context].sum(1)
        
        # r ~ p(r|x,a,e) = p(r|x,e)
        reward = self.random_.normal(loc=expected_reward_factual, scale=self.reward_noise)

        pscore_dict = self.aggregate_propensity_score(
            pi=pi_b,
            action=action,
            p_e_d_a=self.p_e_d_a[:, :, self.n_unobserved_cat_dim:],
            action_context=action_context,
        )

        return dict(
            context=context,
            action=action,
            p_e_d_a=self.p_e_d_a[:, :, self.n_unobserved_cat_dim:],
            action_context=action_context[:, self.n_unobserved_cat_dim:],
            reward=reward,
            pscore=pscore_dict,
            expected_reward=q_x_a,
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, evaluation_policy: np.ndarray
    ) -> np.float64:
        return np.average(expected_reward, weights=evaluation_policy, axis=1).mean()

    def aggregate_propensity_score(
        self,
        pi: np.ndarray,
        action: np.ndarray,
        p_e_d_a: np.ndarray,
        action_context: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        
        rounds = np.arange(len(action_context))
        marginal_pscore = np.ones_like(rounds, dtype=float)
        for d in range(p_e_d_a.shape[-1]):
            
            p_e_pi_d = pi @ p_e_d_a[:, :, d]
            marginal_pscore *= p_e_pi_d[rounds, action_context[:, d]]
        
        pscore = pi[rounds, action]

        pscore_dict = dict(action=pscore, category=marginal_pscore, pi=pi)

        return pscore_dict


@dataclass
class ModifiedOpenBanditDataset(OpenBanditDataset):
    @property
    def n_actions(self) -> int:
        return int(self.action.max() + 1)
    
    def pre_process(self) -> None:
        user_cols = self.data.columns.str.contains('user_feature')
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        position = pd.DataFrame(self.position)
        self.action_context = (
            self.item_context.drop(columns=["item_id", "item_feature_0"], axis=1)
            .apply(LabelEncoder().fit_transform)
            .values
        )
        self.action_context = self.action_context[self.action]
        self.action_context = np.c_[self.action_context, position]

        self.action = self.position * self.n_actions + self.action
        self.position = np.zeros_like(self.position)
        self.pscore /= (self.position.max() + 1)
    
    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:

        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]

        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in [
            "action",
            "position",
            "reward",
            "pscore",
            "context",
            "action_context",
        ]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback
