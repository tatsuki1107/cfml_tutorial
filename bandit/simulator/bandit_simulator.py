from dataclasses import dataclass
from typing import Union, Optional, List, Tuple

import numpy as np

from simulator import ContextualWebServer
from policy import LinThompsonSampling, LinUCB, LogisticThompsonSampling


def _chack_instance(policy, web_server):

    valid_policy = (LinThompsonSampling, LinUCB, LogisticThompsonSampling)
    if not isinstance(policy, valid_policy) and policy != "Random":
        raise ValueError("policy must be a contextual bandit or Random.")

    if not isinstance(web_server, ContextualWebServer):
        raise ValueError("web_server must be a ContextualWebServer")


@dataclass
class BanditSimulator:
    T: int
    batch_size: int
    n_action: int
    action_contexts: Optional[np.ndarray] = None

    def _initialize_data(self) -> None:

        # log data
        # [(context, action, reward, regret), ...]
        self.log_data = []
        self.batch_data = []

    def run(
        self,
        policy: Union[LinThompsonSampling, LinUCB, LogisticThompsonSampling],
        web_server: ContextualWebServer,
    ) -> dict:
        """Run the bandit algorithm.

        Args:
            policy: contextual bandit algorithm
            web_server (ContextualWebServer): web server must be a contextual bandit

        Returns:
            dict: The cumulative reward and cumulative regret.
        """

        _chack_instance(policy, web_server)
        self._initialize_data()

        for t in range(1, self.T + 1):
            # 時刻t時点でweb上に訪れたユーザー文脈をサーバー側で取得
            user_context = web_server.request(t)
            # アクションの文脈との交互作用を考慮した文脈 "contexts" を生成
            contexts = self._preprocess_context(user_context)

            # アクションを選択して、クライアントに返す
            if policy == "Random":
                selected_action = np.random.randint(self.n_action)
            else:
                selected_action = policy.select_action(contexts, t)
            # すぐに報酬がサーバー側に返ってくる
            reward, regret = web_server.response(contexts, selected_action)

            self.batch_data.append(
                (contexts[selected_action], selected_action, reward, regret)
            )

            if t % self.batch_size == 0 or t == self.T:
                if policy != "Random":
                    policy.update_parameter(self.batch_data)
                self.log_data.extend(self.batch_data)
                self.batch_data = []

        contexts, actions, rewards, regrets = map(np.array, zip(*self.log_data))

        cumulative_reward, cumulative_regret = self._calc_cumulative_reward_and_regret(
            rewards, regrets
        )

        self.observed_data = dict(
            context=contexts,
            action=actions,
            reward=rewards,
        )

        return dict(
            reward=rewards,
            cumulative_reward=cumulative_reward,
            cumulative_regret=cumulative_regret,
        )

    def _preprocess_context(self, user_context: np.ndarray) -> np.ndarray:
        if self.action_contexts is None:
            contexts = np.tile(user_context, (self.n_action, 1))

        else:
            contexts = []
            for action_context in self.action_contexts:
                interaction_vector = np.array(
                    [u * a for a in action_context for u in user_context]
                )
                context = np.r_[
                    action_context, user_context, interaction_vector
                ].tolist()
                contexts.append(context)

            contexts = np.array(contexts)

        return contexts

    def _calc_cumulative_reward_and_regret(
        self, rewards: np.ndarray, regrets: np.ndarray
    ) -> Tuple[List[np.float64]]:
        """Calculate the cumulative reward.

        Returns:
            List[np.float64]: The cumulative reward.
        """

        cumulative_reward, cumulative_regret = [0], [0]
        for reward, regret in zip(rewards, regrets):
            curr_sum = cumulative_reward[-1] + reward
            cumulative_reward.append(curr_sum)

            curr_sum = cumulative_regret[-1] + regret
            cumulative_regret.append(curr_sum)

        return cumulative_reward, cumulative_regret
