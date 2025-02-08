from dataclasses import dataclass
from typing import List, Optional

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from ope.estimator import InversePropensityScore
from ope.estimator_tuning import BaseOffPolicyEstimatorWithTune
from ope.importance_weight import vanilla_weight
from ope.importance_weight import marginal_weight_over_embedding_spaces
from ope.importance_weight import marginal_weight_over_cluster_spaces
from ope.importance_weight import estimated_marginal_weight


@dataclass
class OffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: List[InversePropensityScore]
    ope_estimators_tune: Optional[List[BaseOffPolicyEstimatorWithTune]] = None

    def __post_init__(self) -> None:
        self.estimator_names = set(
            [estimator.estimator_name for estimator in self.ope_estimators]
        )
        self.is_model_dependent = False

        if any(name in self.estimator_names for name in ["DM", "DR", "OFFCEM"]):
            self.is_model_dependent = True

    def _create_estimator_inputs(
        self, action_dist: np.ndarray, estimated_rewards: Optional[dict[str, np.ndarray]] = None
    ) -> dict:
        if ("IPS" in self.estimator_names) or ("DR" in self.estimator_names):
            w_x_a = vanilla_weight(data=self.bandit_feedback, action_dist=action_dist)

        if "MIPS (true)" in self.estimator_names or "MDR (true)" in self.estimator_names:
            w_x_e = marginal_weight_over_embedding_spaces(
                data=self.bandit_feedback, 
                action_dist=action_dist
            )

        if ("MIPS" in self.estimator_names) or ("MDR" in self.estimator_names):
            w_x_e_hat = estimated_marginal_weight(
                data=self.bandit_feedback,
                action_dist=action_dist,
                weight_estimator=RandomForestClassifier(n_estimators=10, max_depth=10),
            )
        
        if any("OFFCEM" in name for name in self.estimator_names):
            w_x_c = marginal_weight_over_cluster_spaces(
                data=self.bandit_feedback, 
                action_dist=action_dist
            )

        input_data = {}
        for estimator_name in self.estimator_names:
            input_data_ = {}
            # reward
            input_data_["reward"] = self.bandit_feedback["reward"]

            # weight
            if estimator_name in ["IPS", "DR"]:
                input_data_["weight"] = w_x_a
            elif estimator_name in ["MIPS (true)", "MDR (true)"]:
                input_data_["weight"] = w_x_e
            elif estimator_name in ["MIPS", "MDR"]:
                input_data_["weight"] = w_x_e_hat

            # estimated reward
            if estimator_name == "DR":
                q_hat = estimated_rewards[estimator_name]
                input_data_["q_hat_factual"] = q_hat[:, self.bandit_feedback["action"]]

            if estimator_name in ["DM", "DR"]:
                input_data_["q_hat"] = estimated_rewards[estimator_name]
                input_data_["action_dist"] = action_dist
            
            if "OFFCEM" in estimator_name:
                f_hat = estimated_rewards[estimator_name]
                input_data_["f_hat_factual"] = f_hat[:, self.bandit_feedback["action"]]
                input_data["f_hat"] = f_hat
                input_data["action_dist"] = action_dist

            input_data[estimator_name] = input_data_

        return input_data

    def estimate_policy_values(
        self, action_dist: np.ndarray, estimated_rewards: Optional[dict[str, np.ndarray]] = None
    ) -> dict:
        if (estimated_rewards is None) and self.is_model_dependent:
            raise ValueError("estimated_rewards must be given")

        input_data = self._create_estimator_inputs(
            action_dist=action_dist, estimated_rewards=estimated_rewards
        )

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(
                **input_data[estimator.estimator_name]
            )
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        if self.ope_estimators_tune is not None:
            for estimator_tune in self.ope_estimators_tune:
                estimated_policy_value = estimator_tune.estimate_policy_value_with_tune(
                    bandit_feedback=self.bandit_feedback, action_dist=action_dist
                )
                estimated_policy_values[
                    estimator_tune.estimator.estimator_name
                ] = estimated_policy_value

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


PALETTE = {
    "IPS": "tab:red",
    "DR": "tab:blue",
    "MIPS (true)": "tab:orange",
    "MIPS (true)-SLOPE": "tab:green",
    "MIPS": "tab:gray",
    "AVG": "tab:blue",
}


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
            palette=PALETTE,
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


def visualize_cdf_of_ralative_error(
    rel_result_df: DataFrame, baseline: str = "IPS"
) -> None:
    baseline_se = rel_result_df[rel_result_df["estimator"] == baseline].set_index(
        "index"
    )["se"]
    rel_result_df["baseline_se"] = rel_result_df["index"].map(baseline_se)
    rel_result_df["rel_se"] = rel_result_df["se"] / rel_result_df["baseline_se"]
    rel_result_df = rel_result_df[["estimator", "rel_se"]]
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    sns.ecdfplot(
        linewidth=4,
        palette=PALETTE,
        data=rel_result_df,
        x="rel_se",
        hue="estimator",
        ax=ax,
    )

    # yaxis
    ax.set_ylabel("probability", fontsize=25)
    ax.set_ylim([0, 1.1])
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    # xaxis
    ax.set_xscale("log")
    ax.set_xlabel(f"relative squared errors w.r.t. {baseline}", fontsize=25)
    ax.tick_params(axis="x", labelsize=18)
    ax.xaxis.set_label_coords(0.5, -0.1)
