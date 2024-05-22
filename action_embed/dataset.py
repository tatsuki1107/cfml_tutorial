from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.utils import check_random_state
from obp.utils import softmax, sample_action_fast
from obp.dataset import linear_behavior_policy, logistic_reward_function


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
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

    def obtain_batch_bandit_feedback(
        self, n_rounds: int, return_marginal_pi_b: bool
    ) -> dict:

        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        pi_b_logits = linear_behavior_policy(
            context=context,
            action_context=np.eye(self.n_actions),
            random_state=self.random_state,
        )
        pi_b = softmax(pi_b_logits)
        # a ~ \pi_b(\cdot|x)
        action = sample_action_fast(pi_b, random_state=self.random_state)
        pscore = pi_b[np.arange(n_rounds), action].copy()

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

        # r ~ p(r|x,a,e)
        reward = self.random_.normal(q_x_a_e[np.arange(n_rounds), action])

        if return_marginal_pi_b:
            marginal_pi_b, marginal_pscore = self.compute_marginal_probability(
                pi=pi_b,
                action=action,
                p_e_a=p_e_a,
                category=category,
            )
        else:
            marginal_pi_b, marginal_pscore = None, None

        return dict(
            context=context,
            action=action,
            category=category,
            reward=reward,
            pscore=pscore,
            marginal_pscore=marginal_pscore,
            expected_reward=q_x_a_e,
            p_e_a=p_e_a,
            pi_b=pi_b,
            marginal_pi_b=marginal_pi_b,
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, evaluation_policy: np.ndarray
    ) -> np.float64:
        return np.average(expected_reward, weights=evaluation_policy, axis=1).mean()

    def compute_marginal_probability(
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
        marginal_pscore = pi[np.arange(n_rounds), category[action]].copy()

        return marginal_pi, marginal_pscore
