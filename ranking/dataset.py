from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple
from multiprocessing import Pool, cpu_count
from itertools import permutations

import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import PolynomialFeatures
from obp.utils import sigmoid, softmax, sample_action_fast


class BaseBanditDataset(metaclass=ABCMeta):

    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        raise NotImplementedError


@dataclass
class SyntheticSlateBanditDataset(BaseBanditDataset):
    n_unique_action: int
    len_list: int
    dim_context: int
    reward_structure: str = "independent"
    base_reward_function: Callable = None
    reward_function: Callable = None
    behavior_policy_function: Callable = None
    click_model: str = None
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.action_context = np.eye(self.n_unique_action, dtype=int)
        self.all_slate_actions = np.array(
            list(permutations(range(self.n_unique_action), self.len_list))
        )

        self.behavior_policy_function = mf_behavior_policy_logit
        self.base_reward_function = logistic_reward_function
        self.reward_function = action_interaction_reward_function

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        return_pscore_cascade: bool = False,
    ) -> dict:

        # context ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        behavior_policy_logit_ = self.behavior_policy_function(
            context=context,
            n_unique_action=self.n_unique_action,
            random_state=self.random_state,
        )
        # \pi_b(\mathbb{a} | x)
        ranking_pi_b = self.compute_ranking_pi_given_policy_logit(
            policy_logit=behavior_policy_logit_
        )

        # \mathbb{a} ~ \pi_b(\mathbb{a} | x)
        sampled_slate_index = sample_action_fast(action_dist=ranking_pi_b)
        slate_actions = self.all_slate_actions[sampled_slate_index]
        pscore = ranking_pi_b[np.arange(n_rounds), sampled_slate_index]

        expected_reward_factual = self.reward_function(
            context=context,
            action_context=self.action_context,
            slate_action=slate_actions,
            base_reward_function=self.base_reward_function,
            reward_structure=self.reward_structure,
            len_list=self.len_list,
            random_=self.random_,
        )

        # \mathbb{r} ~ p(\mathbb{r} | x, \mathbb{a})
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual
        )

        # marginalization
        if return_pscore_item_position:
            marginal_pi_b_at_position = self.compute_marginal_probability(
                ranking_pi=ranking_pi_b
            )
            pscore_item_position = marginal_pi_b_at_position[
                np.arange(n_rounds)[:, np.newaxis],
                np.arange(self.len_list),
                slate_actions,
            ]
        else:
            marginal_pi_b_at_position, pscore_item_position = None, None

        if return_pscore_cascade:
            raise NotImplementedError
        else:
            marginal_pi_b_cascade, pscore_cascade = None, None

        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_unique_action,
            context=context,
            action_context=self.action_context,
            action=slate_actions,
            reward=reward,
            expected_reward_factual=expected_reward_factual,
            pscore=pscore,
            pscore_item_position=pscore_item_position,
            pscore_cascade=pscore_cascade,
            ranking_pi_b=ranking_pi_b,
            marginal_pi_b_at_position=marginal_pi_b_at_position,
            marginal_pi_b_cascade=marginal_pi_b_cascade,
        )

    def sample_reward_given_expected_reward(
        self,
        expected_reward_factual: np.ndarray,
    ) -> np.ndarray:

        if self.reward_structure == "independent" and self.click_model is None:
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        else:
            raise NotImplementedError

        return reward

    def compute_marginal_probability(
        self, ranking_pi: np.ndarray, num_workers=cpu_count() - 1
    ):
        with Pool(num_workers) as p:
            job_args = [(ranking_pi_i) for ranking_pi_i in ranking_pi]
            marginal_pi = list(p.imap(self._compute_marginal_probability_i, job_args))

        return np.array(marginal_pi)

    def _compute_marginal_probability_i(self, ranking_pi_i: np.ndarray):

        marginal_pi_i = [[] for _ in range(self.len_list)]
        for pos_ in range(self.len_list):
            for action in range(self.n_unique_action):
                action_indicator = self.all_slate_actions[:, pos_] == action
                marginal_pscore = ranking_pi_i[action_indicator].sum()
                marginal_pi_i[pos_].append(marginal_pscore)

        return marginal_pi_i

    def compute_ranking_pi_given_policy_logit(
        self,
        policy_logit: np.ndarray,
        num_worker: int = cpu_count() - 1,
    ) -> np.ndarray:

        with Pool(num_worker) as p:
            job_args = [policy_logit_i for policy_logit_i in policy_logit]
            ranking_pi = list(p.imap(self._compute_ranking_pi_i, job_args))

        return np.array(ranking_pi)

    def _compute_ranking_pi_i(self, policy_logit_i_):

        n_slate_actions_ = len(self.all_slate_actions)
        unique_action_set_2d = np.tile(
            np.arange(self.n_unique_action), reps=(n_slate_actions_, 1)
        )

        ranking_pi_i = np.ones(n_slate_actions_)
        for pos_ in range(self.len_list):
            indicator = (
                unique_action_set_2d == self.all_slate_actions[:, pos_][:, np.newaxis]
            )
            action_index = np.where(indicator)[1]
            ranking_pi_i *= softmax(policy_logit_i_[unique_action_set_2d])[
                np.arange(n_slate_actions_), action_index
            ]

            if pos_ + 1 != self.len_list:
                mask = np.ones((n_slate_actions_, self.n_unique_action - pos_))
                mask[np.arange(n_slate_actions_), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - pos_ - 1)
                )

        return ranking_pi_i


def mf_behavior_policy_logit(
    context: np.ndarray, n_unique_action: int, random_state=12345, tau=1.0
):
    random_ = check_random_state(random_state)
    action_coef_ = random_.normal(size=(n_unique_action, context.shape[1]))

    logits = context @ action_coef_.T
    return logits / tau


def sample_random_uniform_coefficients(
    dim_context: int, dim_action_context: int, random_: np.random.RandomState
) -> Tuple[np.ndarray, ...]:
    context_coef_ = random_.uniform(-1, 1, size=dim_context)
    action_coef_ = random_.uniform(-1, 1, size=dim_action_context)
    context_action_coef_ = random_.uniform(
        -1, 1, size=(dim_context, dim_action_context)
    )

    return context_coef_, action_coef_, context_action_coef_


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_: np.random.RandomState,
    coef_function: Callable = sample_random_uniform_coefficients,
    degree: int = 1,
    z_score: bool = True,
) -> np.ndarray:

    poly = PolynomialFeatures(degree=degree)
    context_ = poly.fit_transform(context)
    action_context_ = poly.fit_transform(action_context)

    dim_context = context_.shape[1]
    n_action, dim_action_context = action_context_.shape

    context_coef_, action_coef_, context_action_coef_ = coef_function(
        dim_context=dim_context, dim_action_context=dim_action_context, random_=random_
    )

    context_values = np.tile(context_ @ context_coef_, reps=(n_action, 1)).T
    action_values = action_coef_ @ action_context_.T
    context_action_values = context_ @ context_action_coef_ @ action_context_.T

    expected_rewards = context_values + action_values + context_action_values

    if z_score:
        expected_rewards = (
            expected_rewards - expected_rewards.mean()
        ) / expected_rewards.std()

    return sigmoid(expected_rewards)


def action_interaction_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    slate_action: np.ndarray,
    base_reward_function: Callable,
    reward_structure: str,
    len_list: int,
    random_: np.random.RandomState,
) -> np.ndarray:

    expected_reward = base_reward_function(
        context=context,
        action_context=action_context,
        random_=random_,
    )
    n_rounds = len(context)

    expected_reward_factual = []
    for pos_ in range(len_list):
        if reward_structure == "independent":
            expected_reward_pos_ = expected_reward[
                np.arange(n_rounds), slate_action[:, pos_]
            ]
        else:
            raise NotImplementedError
        expected_reward_factual.append(expected_reward_pos_)

    expected_reward_factual = np.array(expected_reward_factual).T

    return expected_reward_factual
