from dataclasses import dataclass
from typing import List, Optional

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from estimator import InversePropensityScore
from estimator_tuning import BaseOffPolicyEstimatorWithTune
from importance_weight import vanilla_weight
from importance_weight import marginal_weight
from importance_weight import estimated_marginal_weight


@dataclass
class ActionEmbedOffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[InversePropensityScore]
    ope_estimators_tune: Optional[List[BaseOffPolicyEstimatorWithTune]] = None
    
    def __post_init__(self) -> None:
        self.estimator_names = set([estimator.estimator_name for estimator in self.ope_estimators])

    def _create_estimator_inputs(self, action_dist: np.ndarray) -> dict:
        
        
        if ("IPS" in self.estimator_names) or ("DR" in self.estimator_names):
            w_x_a = vanilla_weight(data=self.bandit_feedback, action_dist=action_dist)
        
        if ("MIPS (true)" in self.estimator_names) or ("MDR (true)" in self.estimator_names):
            w_x_e = marginal_weight(data=self.bandit_feedback, action_dist=action_dist)
        
        if ("MIPS" in self.estimator_names) or ("MDR" in self.estimator_names):
            w_x_e_hat = estimated_marginal_weight(
                data=self.bandit_feedback, 
                action_dist=action_dist,
                weight_estimator=RandomForestClassifier(n_estimators=10, max_depth=10)
            )

        input_data = {}
        for estimator_name in self.estimator_names:
            input_data[estimator_name] = {"reward": self.bandit_feedback["reward"]}
            
            if estimator_name in ["IPS", "DR"]:
                input_data[estimator_name]["weight"] = w_x_a
            elif estimator_name in ["MIPS (true)", "MDR (true)"]:
                input_data[estimator_name]["weight"] = w_x_e
            elif estimator_name in ["MIPS", "MDR"]:
                input_data[estimator_name]["weight"] = w_x_e_hat


        return input_data

    def estimate_policy_values(self, action_dist: np.ndarray) -> dict:

        input_data = self._create_estimator_inputs(action_dist=action_dist)

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(
                **input_data[estimator.estimator_name]
            )
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value
        
        if self.ope_estimators_tune is not None:
            for estimator_tune in self.ope_estimators_tune:
                estimated_policy_value = estimator_tune.estimate_policy_value_with_tune(
                    bandit_feedback=self.bandit_feedback,
                    action_dist=action_dist
                )
                estimated_policy_values[estimator_tune.estimator.estimator_name] = estimated_policy_value

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
    palette = {
        "IPS": "tab:red", 
        "MIPS (true)": "tab:orange", 
        "MIPS (true)-SLOPE": "tab:green",
        "MIPS": "tab:gray", 
        "AVG": "tab:blue"
    }

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
