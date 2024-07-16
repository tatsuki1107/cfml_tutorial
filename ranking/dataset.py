from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional, Dict
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
    behavior_ratio: dict
    base_reward_function: Callable = None
    reward_function: Callable = None
    behavior_policy_function: Callable = None
    random_state: int = 12345
    reward_noise: float = 1.0
    interaction_noise: float = 1.0

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.action_context = np.eye(self.n_unique_action, dtype=int)
        self.all_slate_actions = np.array(
            list(permutations(range(self.n_unique_action), self.len_list))
        )

        self.user_behavior_to_index = {
            behabior_name: i
            for i, behabior_name in enumerate(self.behavior_ratio.keys())
        }

        self.behavior_policy_function = mf_behavior_policy_logit
        self.base_reward_function = logistic_reward_function
        self.reward_function = action_interaction_reward_function
        
        self.interaction_params = self.random_.normal(
            scale=self.interaction_noise, size=(self.len_list, self.len_list)
        )

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore: Dict[str, bool],
        return_pi_b: bool = False,
    ) -> dict:
        # x ~ p(x)
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
        sampled_slate_index = sample_action_fast(
            action_dist=ranking_pi_b
        )

        slate_actions = self.all_slate_actions[sampled_slate_index]

        # \mathbb{c} ~ p(\mathbb{c} | x)
        user_behavior_idx, user_behavior = self._sample_user_behavior(
            n_rounds=n_rounds
        )

        expected_reward = self.reward_function(
            context=context,
            action_context=self.action_context,
            all_slate_action=self.all_slate_actions,
            base_reward_function=self.base_reward_function,
            user_behavior_to_index=self.user_behavior_to_index,
            interaction_params=self.interaction_params,
            random_state=self.random_state,
        )

        # \mathbb{r} ~ p(\mathbb{r} | x, \mathbb{a})
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward[
                np.arange(n_rounds), user_behavior_idx, sampled_slate_index
            ]
        )

        pi_b, pscore = self.aggregate_propensity_score(
            ranking_pi=ranking_pi_b,
            slate_id=sampled_slate_index,
            return_pscore=return_pscore,
            user_behavior=user_behavior,
            return_pi=return_pi_b,
        )

        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_unique_action,
            slate_id=sampled_slate_index,
            context=context,
            action_context=self.action_context,
            action=slate_actions,
            reward=reward,
            expected_reward=expected_reward,
            user_behavior_id=user_behavior_idx,
            user_behavior=user_behavior,
            pscore=pscore,
            pi_b=pi_b,
        )

    def calc_ground_truth_policy_value(
        self,
        alpha: np.ndarray,
        expected_reward: np.ndarray,
        evaluation_policy: np.ndarray,
        user_behavior_prob: np.ndarray,
    ) -> np.float64:

        # \sum_{k=1}^K \alpha_k q_k(x, \Phi_k(\mathbb{c}, \mathbb{a}))
        weighted_expected_reward = np.sum(expected_reward * alpha, axis=3)

        # \sum_{\mathbb{a}} \pi_e(\mathbb{a} | x) * weighted_expected_reward
        policy_weighted_reward = np.sum(
            weighted_expected_reward * evaluation_policy[:, np.newaxis, :], axis=2
        )

        # \frac{1}{n} \sum_{i=1}^n \sum_{\mathbb{c}} p(\mathbb{c}|x) *
        # policy_weighted_reward
        return np.average(
            policy_weighted_reward, weights=user_behavior_prob, axis=1
        ).mean()

    def _sample_user_behavior(
        self, n_rounds: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # sample context free behavior
        user_behavior = self.random_.choice(
            list(self.behavior_ratio.keys()),
            p=list(self.behavior_ratio.values()),
            size=n_rounds,
        )
        user_behabior_idx = np.array(
            [
                self.user_behavior_to_index[behabior_name]
                for behabior_name in user_behavior
            ]
        )

        return user_behabior_idx, user_behavior

    def sample_reward_given_expected_reward(
        self,
        expected_reward_factual: np.ndarray,
    ) -> np.ndarray:

        reward = self.random_.normal(expected_reward_factual, scale=self.reward_noise)
        return reward

    def compute_marginal_probability_at_position(
        self,
        ranking_pi: np.ndarray,
        slate_actions: np.ndarray,
        num_workers=cpu_count() - 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with Pool(num_workers) as p:
            job_args = [(ranking_pi_i) for ranking_pi_i in ranking_pi]
            marginal_pi = list(
                p.imap(self._compute_marginal_probability_at_position_i, job_args)
            )

        marginal_pi = np.array(marginal_pi)

        n_rounds = marginal_pi.shape[0]
        pscore_item_position = marginal_pi[
            np.arange(n_rounds)[:, np.newaxis],
            np.arange(self.len_list),
            slate_actions,
        ]

        return marginal_pi, pscore_item_position

    def _compute_marginal_probability_at_position_i(
        self, ranking_pi_i: np.ndarray
    ) -> list:
        marginal_pi_i = [[] for _ in range(self.len_list)]
        for pos_ in range(self.len_list):
            for action in range(self.n_unique_action):
                action_indicator = self.all_slate_actions[:, pos_] == action
                marginal_pscore = ranking_pi_i[action_indicator].sum()
                marginal_pi_i[pos_].append(marginal_pscore)

        return marginal_pi_i

    def compute_marginal_probability_cascade(
        self,
        ranking_pi: np.ndarray,
        slate_actions: np.ndarray,
        num_workers=cpu_count() - 1,
    ) -> Tuple[List[dict], np.ndarray]:
        with Pool(num_workers) as p:
            job_args = [(ranking_pi_i) for ranking_pi_i in ranking_pi]
            marginal_pi = list(
                p.imap(self._compute_marginal_probability_cascade_i, job_args)
            )

        marginal_pscore = []
        for marginal_pi_i, action in zip(marginal_pi, slate_actions):
            pscore_1_to_k = [
                marginal_pi_i[tuple(action[: pos_ + 1])]
                for pos_ in range(self.len_list)
            ]
            marginal_pscore.append(pscore_1_to_k)

        marginal_pscore = np.array(marginal_pscore)

        return marginal_pi, marginal_pscore

    def _compute_marginal_probability_cascade_i(self, ranking_pi_i: np.ndarray) -> dict:
        marginal_pi_i = {}
        for pos_ in range(self.len_list):
            for slate in np.array(
                list(permutations(range(self.n_unique_action), pos_ + 1))
            ):
                action_indicator = np.all(
                    self.all_slate_actions[:, : pos_ + 1] == slate[: pos_ + 1], axis=1
                )
                marginal_pscore = ranking_pi_i[action_indicator].sum()
                marginal_pi_i[tuple(slate)] = marginal_pscore

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

    def aggregate_propensity_score(
        self,
        ranking_pi: np.ndarray,
        slate_id: np.ndarray,
        return_pscore: Dict[str, bool],
        user_behavior: np.ndarray,
        return_pi: bool = False,
    ) -> Tuple[Optional[dict], dict]:
        # marginalization
        if return_pscore["independent"]:
            (
                marginal_pi_at_position,
                pscore_item_position,
            ) = self.compute_marginal_probability_at_position(
                ranking_pi=ranking_pi,
                slate_actions=self.all_slate_actions[slate_id],
            )
        else:
            marginal_pi_at_position, pscore_item_position = None, None

        if return_pscore["cascade"]:
            (
                marginal_pi_cascade,
                pscore_cascade,
            ) = self.compute_marginal_probability_cascade(
                ranking_pi=ranking_pi,
                slate_actions=self.all_slate_actions[slate_id],
            )

        else:
            marginal_pi_cascade, pscore_cascade = None, None

        n_rounds = len(slate_id)
        pscore_all = ranking_pi[np.arange(n_rounds), slate_id]
        tiled_pscore_all = np.tile(pscore_all[:, np.newaxis], reps=self.len_list)
        pscore = {
            "independent": pscore_item_position,
            "cascade": pscore_cascade,
            "all": tiled_pscore_all,
        }

        if return_pscore["adaptive"]:
            pscore_adaptive = []
            for i, user_behavior in enumerate(user_behavior):
                pscore_adaptive.append(pscore[user_behavior][i])
            pscore_adaptive = np.array(pscore_adaptive)

        else:
            pscore_adaptive = None

        pscore["adaptive"] = pscore_adaptive

        if return_pscore["adaptive"] and return_pi:
            raise NotImplementedError

        if return_pi:
            pi = {
                "independent": marginal_pi_at_position,
                "cascade": marginal_pi_cascade,
                "all": ranking_pi,
            }
        else:
            pi = None

        return pi, pscore


def mf_behavior_policy_logit(
    context: np.ndarray, 
    n_unique_action: int, 
    random_state: Optional[int] = None, 
    tau: float = 1.0
) -> np.ndarray:
    random_ = check_random_state(random_state)
    action_coef_ = random_.normal(size=(n_unique_action, context.shape[1]))

    logits = context @ action_coef_.T
    return logits / tau


def sample_random_uniform_coefficients(
    dim_context: int, dim_action_context: int, random_state: int
) -> Tuple[np.ndarray, ...]:
    random_ = check_random_state(random_state)
    context_coef_ = random_.uniform(-1, 1, size=dim_context)
    action_coef_ = random_.uniform(-1, 1, size=dim_action_context)
    context_action_coef_ = random_.uniform(
        -1, 1, size=(dim_context, dim_action_context)
    )

    return context_coef_, action_coef_, context_action_coef_


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int,
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
        dim_context=dim_context, dim_action_context=dim_action_context, random_state=random_state
    )

    context_values = np.tile(context_ @ context_coef_, reps=(n_action, 1)).T
    action_values = action_coef_ @ action_context_.T
    context_action_values = context_ @ context_action_coef_ @ action_context_.T

    expected_rewards = context_values + action_values + context_action_values

    if z_score:
        expected_rewards = (
            (expected_rewards - expected_rewards.mean())
        ) / expected_rewards.std()

    return sigmoid(expected_rewards)


def action_interaction_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    all_slate_action: np.ndarray,
    base_reward_function: Callable,
    user_behavior_to_index: dict,
    interaction_params: np.ndarray,
    random_state: int
) -> np.ndarray:
    expected_reward_per_action: np.ndarray = base_reward_function(
        context=context,
        action_context=action_context,
        random_state=random_state,
    )
    n_rounds = len(context)

    # shape: (n_rounds, n_slate, len_list)

    expected_reward = expected_reward_per_action[
        np.arange(n_rounds)[:, np.newaxis], all_slate_action[:, np.newaxis]
    ].transpose(1, 0, 2)

    # shape: (n_rounds, n_user_behavior, n_slate, len_list)
    expected_reward = np.tile(
        expected_reward[:, np.newaxis, :, :],
        reps=(1, len(user_behavior_to_index), 1, 1),
    )

    expected_reward_fixed = expected_reward.copy()

    for behavior, c in user_behavior_to_index.items():

        if behavior == "independent":
            continue

        for pos_ in range(len(interaction_params)):
            if behavior == "cascade":
                expected_reward_fixed[:, c, :, pos_] += (
                    expected_reward[:, c, :, :pos_] * interaction_params[pos_, :pos_]
                ).sum(axis=2)

            elif behavior == "all":
                expected_reward_fixed[:, c, :, pos_] += (
                    expected_reward[:, c, :, :pos_] * interaction_params[pos_, :pos_]
                ).sum(axis=2)
                expected_reward_fixed[:, c, :, pos_] += (
                    expected_reward[:, c, :, pos_ + 1 :]
                    * interaction_params[pos_, pos_ + 1 :]
                ).sum(axis=2)

    return expected_reward_fixed
