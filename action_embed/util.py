from dataclasses import dataclass
from typing import List

from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from estimator import InversePropensityScore


@dataclass
class ActionEmbedOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[InversePropensityScore]
    estimator_to_pscore_dict: dict

    def _create_estimator_inputs(
        self,
        evaluation_policy_pscore: dict,
    ) -> dict:

        input_data = {}
        for estimator_name, pscore_name in self.estimator_to_pscore_dict.items():
            input_data[estimator_name] = {
                "reward": self.bandit_feedback["reward"],
                "behavior_policy_pscore": self.bandit_feedback["pscore"][pscore_name],
                "evaluation_policy_pscore": evaluation_policy_pscore[pscore_name],
            }

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
