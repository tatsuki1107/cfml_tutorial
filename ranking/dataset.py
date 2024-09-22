from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state
from obp.utils import softmax
from obp.utils import sample_action_fast
from obp.dataset import BaseBanditDataset
from obp.dataset import logistic_reward_function

from utils.policy import gen_eps_greedy
from utils.sampling import sample_ranking_fast_with_replacement
from utils.user_behavior import create_interaction_params


@dataclass
class SyntheticRankingBanditDataset(BaseBanditDataset):
    n_actions_at_k: int
    len_list: int
    dim_context: int
    behavior_params: dict
    beta: float
    eps: float
    is_replacement: bool = True
    random_state: int = 12345
    reward_noise: float = 1.0
    interaction_noise: float = 1.0
    delta: float = 1.0

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.n_actions = self.n_actions_at_k * self.len_list
        self.action_context = np.eye(self.n_actions, dtype=int)

        self.interaction_params = create_interaction_params(
            behavior_names=list(self.behavior_params.keys()),
            len_list=self.len_list,
            interaction_noise=self.interaction_noise,
            random_state=self.random_state,
        )

        if self.is_replacement:
            self.candidate_action_set_at_k = np.arange(self.n_actions).reshape(
                self.len_list, self.n_actions_at_k
            )
        else:
            raise NotImplementedError

        if len(self.behavior_params) > 1:
            self.gamma_z = np.array(list(self.behavior_params.values()))
        else:
            self.gamma_z = None

        self.id2user_behavior = {
            i: c for i, (c, _) in enumerate(self.behavior_params.items())
        }

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
        p_c_x = linear_user_behavior_model(
            context=context,
            gamma_z=self.gamma_z,
            delta=self.delta,
            random_state=self.random_state,
        )
        user_behavior_id = sample_action_fast(p_c_x)
        user_behavior = np.array([self.id2user_behavior[i] for i in user_behavior_id])

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
    interaction_params: dict[str, np.ndarray],
) -> np.ndarray:
    len_list = base_expected_reward_factual.shape[1]

    expected_reward_factual_fixed = np.zeros_like(base_expected_reward_factual)
    for behavior_name in np.unique(user_behavior):
        behavior_mask = user_behavior == behavior_name

        for pos_ in range(len_list):
            q_x_a_k_fixed = (
                interaction_params[behavior_name][pos_]
                * base_expected_reward_factual[behavior_mask]
            ).sum(1)
            expected_reward_factual_fixed[behavior_mask, pos_] = q_x_a_k_fixed

    return expected_reward_factual_fixed


def linear_user_behavior_model(
    context: np.ndarray,
    gamma_z: Optional[np.ndarray],
    delta: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    if gamma_z is None:
        n_rounds = len(context)
        p_c_x = np.ones((n_rounds, 1))
    else:
        random_ = check_random_state(random_state)
        n_behavior_model, dim_context = len(gamma_z), context.shape[1]
        user_behavior_coef = random_.uniform(
            -1, 1, size=(dim_context, n_behavior_model)
        )

        behavior_logits = np.abs(context @ user_behavior_coef)
        lambda_z = np.exp((2 * delta - 1) * gamma_z)
        p_c_x = softmax(lambda_z * behavior_logits)

    return p_c_x
