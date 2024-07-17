from dataclasses import dataclass
from typing import List

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from estimator import InversePropensityScore


@dataclass
class ActionEmbedOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[InversePropensityScore]
    estimator_to_pscore_dict: dict
    
    def _estimate_w_x_e(
        self, 
        pi_e: np.ndarray,
        max_iter: int = 1000,
        random_state: int = 12345,
    ) -> None:
        
        encoder = OneHotEncoder(sparse=False, drop="first")
        onehot_action_context = encoder.fit_transform(
            self.bandit_feedback["action_context"]
        )
        x_e = np.c_[self.bandit_feedback["context"], onehot_action_context]
        
        pi_a_x_e_estimator = LogisticRegression(max_iter=max_iter, random_state=random_state)
        pi_a_x_e_estimator.fit(x_e, self.bandit_feedback["action"])

        
        w_x_a = pi_e / self.bandit_feedback["pscore"]["pi"]
        pi_a_x_e_hat = np.zeros_like(w_x_a)
        pi_a_x_e_hat[:, np.unique(self.bandit_feedback["action"])] = pi_a_x_e_estimator.predict_proba(x_e)
        w_x_e_hat = (w_x_a * pi_a_x_e_hat).sum(axis=1)
        
        return w_x_e_hat

    def _create_estimator_inputs(
        self,
        evaluation_policy_pscore: dict,
    ) -> dict:
        
        if "estimated_category" in set(self.estimator_to_pscore_dict.values()):
            w_x_e_hat = self._estimate_w_x_e(pi_e=evaluation_policy_pscore["pi"])

        input_data = {}
        for estimator_name, pscore_name in self.estimator_to_pscore_dict.items():
            input_data[estimator_name] = {"reward": self.bandit_feedback["reward"]}
            
            if pscore_name == "estimated_category":
                input_data[estimator_name]["weight"] = w_x_e_hat
            else:
                w_x_e = evaluation_policy_pscore[pscore_name] / self.bandit_feedback["pscore"][pscore_name]
                input_data[estimator_name]["weight"] = w_x_e

        return input_data

    def estimate_policy_values(
        self,
        evaluation_policy_pscore: dict,
    ) -> dict:

        input_data = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore
        )

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(
                **input_data[estimator.estimator_name]
            )
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values


def aggregate_simulation_results(
    simulation_result_list: list,
    policy_value: float,
    x_value: int,
) -> DataFrame:
    result_df = (
        DataFrame(DataFrame(simulation_result_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "estimator", 0: "value"})
    )
    result_df["x"] = x_value
    result_df["se"] = (result_df["value"] - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0

    expected_values = (
        result_df.groupby("estimator").agg({"value": "mean"})["value"].to_dict()
    )
    for estimator_name, expected_value in expected_values.items():

        row = result_df["estimator"] == estimator_name

        result_df.loc[row, "bias"] = (policy_value - expected_value) ** 2

        estimated_values = result_df[row]["value"].values
        result_df.loc[row, "variance"] = estimated_values.var()

    return result_df


def visualize_mean_squared_error(result_df: DataFrame, xlabel: str) -> None:

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    y = ["se", "bias", "variance"]
    title = ["mean squared error (MSE)", "Squared Bias", "Variance"]
    palette = {"IPS": "tab:red", "MIPS (true)": "tab:orange", "MIPS": "tab:gray"}

    ylims = []

    for ax_, y_, title_ in zip(axes, y, title):

        sns.lineplot(
            data=result_df,
            x="x",
            y=y_,
            hue="estimator",
            marker="o",
            ci=None,
            markersize=20,
            ax=ax_,
            palette=palette,
        )

        # title
        ax_.set_title(title_, fontsize=25)
        # yaxis
        ax_.set_ylabel("")
        # xaxis
        ax_.set_xlabel(xlabel, fontsize=18)

        if y_ == "se":
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    # 最初のプロットのY軸範囲を他のすべてのサブプロットに適用
    for ax_ in axes:
        ax_.set_ylim(ylims)

    plt.tight_layout()
    plt.show()
