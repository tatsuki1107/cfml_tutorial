from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


PELETTE = {
    "IPS_LogisticRegression": "tab:red",
    "IPS_GradientBoost": "tab:red",
    "Naive_LogisticRegression": "tab:blue",
    "Naive_GradientBoost": "tab:blue",
}
LINESTYLE = {
    "IPS_LogisticRegression": "",
    "IPS_GradientBoost": (4, 2),
    "Naive_LogisticRegression": "",
    "Naive_GradientBoost": (4, 2),
}


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
            palette=PELETTE,
            style="estimator",
            dashes=LINESTYLE,
        )

        # title
        ax_.set_title(title_, fontsize=25)
        # yaxis
        ax_.set_ylabel("")
        # xaxis
        ax_.set_xlabel(xlabel, fontsize=18)
        ax_.set_xticks(result_df["x"].unique())
        ax_.set_xticklabels(result_df["x"].unique(), fontsize=18)

        if y_ == "se":
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    # 最初のプロットのY軸範囲を他のすべてのサブプロットに適用
    for ax_ in axes:
        ax_.set_ylim(ylims)

    plt.tight_layout()
    plt.show()
