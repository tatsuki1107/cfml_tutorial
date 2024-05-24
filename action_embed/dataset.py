from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from sklearn.utils import check_random_state
from obp.utils import softmax, sample_action_fast
from obp.dataset import logistic_reward_function


class BaseBanditDataset(metaclass=ABCMeta):
    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        raise NotImplementedError


@dataclass
class SyntheticBanditDatasetWithActionEmbeds(BaseBanditDataset):

    n_actions: int
    dim_context: int
    n_category: int
    is_category_probabialistic: bool = True
    p_e_a_param_std: float = 1.0
    reward_noise: float = 1.0
    beta: float = -3.0
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:

        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        if self.is_category_probabialistic:
            # e ~ p(e|x,a)
            p_e_a = softmax(
                self.random_.normal(
                    scale=self.p_e_a_param_std, size=(self.n_actions, self.n_category)
                )
            )
            category = sample_action_fast(p_e_a, random_state=self.random_state)
        else:
            NotImplementedError

        latent_cat_param = np.eye(self.n_category)
        q_x_e = logistic_reward_function(
            context=context,
            action_context=latent_cat_param,
            random_state=self.random_state,
        )
        # p(r|x,a,e) = p(r|x,e)
        q_x_a_e = q_x_e[np.arange(n_rounds)[:, np.newaxis], category].copy()

        pi_b = softmax(self.beta * q_x_a_e)
        # a ~ \pi_b(\cdot|x)
        action = sample_action_fast(pi_b, random_state=self.random_state)
        pscore = pi_b[np.arange(n_rounds), action].copy()

        # r ~ p(r|x,a,e)
        reward = self.random_.normal(
            q_x_a_e[np.arange(n_rounds), action], scale=self.reward_noise
        )

        # p(r|x,a) = \sum_{e} p(r|x,a,e) p(e|x,a)
        q_x_a = q_x_e @ p_e_a.T

        pi_b_dict, pscore_dict = self.aggregate_propensity_score(
            pi=pi_b,
            action=action,
            p_e_a=p_e_a,
            category=category,
            pscore=pscore,
        )

        return dict(
            context=context,
            action=action,
            category=category,
            reward=reward,
            pscore=pscore_dict,
            expected_reward=q_x_a,
            p_e_a=p_e_a,
            pi_b=pi_b_dict,
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, evaluation_policy: np.ndarray
    ) -> np.float64:
        return np.average(expected_reward, weights=evaluation_policy, axis=1).mean()

    def aggregate_propensity_score(
        self,
        pi: np.ndarray,
        action: np.ndarray,
        p_e_a: np.ndarray,
        category: np.ndarray,
        pscore: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        marginal_pi, marginal_pscore = self._compute_marginal_probability(
            pi=pi,
            action=action,
            p_e_a=p_e_a,
            category=category,
        )

        pscore_dict = dict(action=pscore, category=marginal_pscore)
        pi_b_dict = dict(action=pi, category=marginal_pi)

        return pi_b_dict, pscore_dict

    def _compute_marginal_probability(
        self,
        pi: np.ndarray,
        action: np.ndarray,
        p_e_a: np.ndarray,
        category: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.is_category_probabialistic:
            marginal_pi = pi @ p_e_a
        else:
            marginal_pi = []
            for c in range(self.n_category):
                marginal_pi.append(pi[:, category == c].sum(1))
            marginal_pi = np.array(marginal_pi).T

        n_rounds = len(pi)
        marginal_pscore = marginal_pi[np.arange(n_rounds), category[action]].copy()

        return marginal_pi, marginal_pscore
