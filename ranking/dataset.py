from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
from obp.utils import softmax
from obp.dataset import BaseBanditDataset
from obp.dataset import logistic_reward_function

from utils.policy import gen_eps_greedy
from utils.sampling import sample_ranking_fast_with_replacement


@dataclass
class SyntheticRankingBanditDataset(BaseBanditDataset):
    n_actions_at_k: int
    len_list: int
    dim_context: int
    behavior_ratio: dict
    beta: float
    eps: float
    is_replacement: bool = True
    random_state: int = 12345
    reward_noise: float = 1.0
    interaction_noise: float = 1.0

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.n_actions = self.n_actions_at_k * self.len_list
        self.action_context = np.eye(self.n_actions, dtype=int)

        self.interaction_params = self.random_.uniform(
            0.0, self.interaction_noise, size=(self.len_list, self.len_list)
        )

        if self.is_replacement:
            self.candidate_action_set_at_k = np.arange(self.n_actions).reshape(
                self.len_list, self.n_actions_at_k
            )
        else:
            raise NotImplementedError

    def obtain_batch_bandit_feedback(
        self, n_rounds: int, is_online: bool = False
    ) -> dict:
        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        # r
        q_x_a = logistic_reward_function(
            context=context,
            action_context=self.action_context,
            random_state=self.random_state,
        )

        if self.is_replacement:
            q_x_a_k = q_x_a[:, self.candidate_action_set_at_k.T]
            pi_b = (
                gen_eps_greedy(
                    q_x_a_k, eps=self.eps, is_replacement=self.is_replacement
                )
                if is_online
                else softmax(self.beta * q_x_a_k)
            )

            action_id_at_k, rankings = sample_ranking_fast_with_replacement(
                pi_b, candidate_action_set_at_k=self.candidate_action_set_at_k
            )
            pscore = pi_b[
                np.arange(n_rounds)[:, None],
                action_id_at_k,
                np.arange(self.len_list)[None, :],
            ]
            evaluation_policy_logit = q_x_a_k

        else:
            raise NotImplementedError

        # c ~ p(c|x)
        user_behavior = self.random_.choice(
            list(self.behavior_ratio.keys()),
            p=list(self.behavior_ratio.values()),
            size=n_rounds,
        )

        rounds = np.arange(n_rounds)[:, None]
        base_expected_reward_factual = q_x_a[rounds, rankings].copy()
        expected_reward_factual = action_interaction_reward_function(
            base_expected_reward_factual=base_expected_reward_factual,
            user_behavior=user_behavior,
            interaction_params=self.interaction_params,
        )
        # r ~ p(r|x,a)
        reward = self.random_.normal(
            loc=expected_reward_factual, scale=self.reward_noise
        )

        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_actions,
            len_list=self.len_list,
            context=context,
            pscore=pscore,
            action=rankings,
            action_context=self.action_context,
            reward=reward,
            pi_b=pi_b,
            evaluation_policy_logit=evaluation_policy_logit,
            expected_reward_factual=expected_reward_factual,
            user_behavior=user_behavior,
            is_replacement=self.is_replacement,
            action_id_at_k=action_id_at_k if self.is_replacement else None,
            ranking_id=None,
            all_rankings=None,
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


def action_interaction_reward_function(
    base_expected_reward_factual: np.ndarray,
    user_behavior: np.ndarray,
    interaction_params: np.ndarray,
) -> np.ndarray:
    position = np.arange(len(interaction_params))
    len_list = len(interaction_params)

    expected_reward_factual_fixed = base_expected_reward_factual.copy()
    expected_reward_factual_fixed /= len_list

    for behavior in np.unique(user_behavior):
        c = user_behavior == behavior

        if behavior == "independent":
            continue

        for pos_ in range(len_list):
            if behavior == "cascade":
                expected_reward_factual_fixed[c, pos_] += (
                    interaction_params[pos_, :pos_]
                    * expected_reward_factual_fixed[c, :pos_]
                    / np.abs(position[:pos_] - pos_)
                ).sum(axis=1)
            elif behavior == "standard":
                expected_reward_factual_fixed[c, pos_] += (
                    interaction_params[pos_, :pos_]
                    * expected_reward_factual_fixed[c, :pos_]
                    / np.abs(position[:pos_] - pos_)
                ).sum(axis=1)
                expected_reward_factual_fixed[c, pos_] += (
                    interaction_params[pos_, pos_ + 1 :]
                    * expected_reward_factual_fixed[c, pos_ + 1 :]
                    / np.abs(position[pos_ + 1 :] - pos_)
                ).sum(axis=1)
            else:
                raise NotImplementedError

    return expected_reward_factual_fixed
