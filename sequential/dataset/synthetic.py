from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
from obp.utils import softmax
from obp.utils import sample_action_fast
from obp.dataset import BaseBanditDataset
from obp.dataset import logistic_reward_function


@dataclass
class SyntheticMDPDataset(BaseBanditDataset):
    """Synthetic data class generation based on the markov decision process (MDP)."""

    H: int
    n_states: int
    dim_state: int
    n_actions: int
    beta: float
    reward_noise: float
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.e_a = np.eye(self.n_actions)

        # initial state distribution
        random_ = check_random_state(self.random_state)
        self.p_s1 = softmax(random_.normal(size=(1, self.n_states)))[
            0
        ]  # shape: (|S_1|,)

        # transition probability
        random_ = check_random_state(self.random_state)
        self.p_s_h1_s_a = softmax(
            random_.normal(size=(self.n_states, self.n_states, self.n_actions))
        )
        # shape: (|S_{h-1}|, |S_h|, |A_{h-1}|)

        # state embedding
        self.S = random_.normal(size=(self.n_states, self.dim_state))
        # imidiate reward function
        self.q_s_a = logistic_reward_function(
            context=self.S, action_context=self.e_a, random_state=self.random_state
        )

        # behavior policy, which is a markov policy based on softmax function.
        self.pi_0_a_s = softmax(self.beta * self.q_s_a)

        self.true_dist_dict = {
            "p_s1": self.p_s1,
            "pi_0_a_s": self.pi_0_a_s,
            "p_s_h1_s_a": self.p_s_h1_s_a,
            "q_s_a": self.q_s_a,
        }

        self.random_ = check_random_state(self.random_state)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> dict:
        states, actions, rewards = [], [], []
        for h in range(self.H):
            if h == 0:
                # s_1 ~ p(s_1)
                p_s1_ = p_s1_ = np.tile(self.p_s1, reps=(n_rounds, 1))
                state = sample_action_fast(p_s1_)
                states.append(state)

            # a_h ~ \pi_0(\cdot|s_h)
            pi_0 = self.pi_0_a_s[state]
            action = sample_action_fast(pi_0)
            actions.append(action)

            # r_h ~ p(r_h|s_h, a_h)
            q_s_a_ = self.q_s_a[state, action]
            reward = self.random_.normal(q_s_a_, scale=self.reward_noise)
            rewards.append(reward)

            if h < self.H - 1:
                # s_h+1 ~ p(s_h+1|s_h, a_h)
                p_s_h1_s_a_ = self.p_s_h1_s_a[state, :, action]
                state = sample_action_fast(p_s_h1_s_a_)
                states.append(state)

        states, actions, rewards = (
            np.array(states).T,
            np.array(actions).T,
            np.array(rewards).T,
        )

        pi_0 = self.pi_0_a_s[states]
        pscores = self.pi_0_a_s[states, actions]

        return dict(
            n_rounds=n_rounds,
            n_states=self.n_states,
            n_actions=self.n_actions,
            state=states,
            action=actions,
            reward=rewards,
            pscore=pscores,
            pi_0=pi_0,
        )

    def calc_ground_truth_policy_value(self, pi: np.ndarray) -> float:
        """calculate the ground-truth policy value of the evaluation policy."""

        if pi.shape != (self.n_states, self.n_actions):
            raise ValueError("The shape of the policy must be (n_states, n_actions).")

        V_ = np.zeros((self.n_states, self.n_actions))
        for _ in range(self.H - 1):
            q_s_pi = (pi * (self.q_s_a + V_)).sum(1)
            V_ = (self.p_s_h1_s_a * q_s_pi[None, :, None]).sum(1)

        V = (self.p_s1[:, None] * pi * (self.q_s_a + V_)).sum()

        return V
