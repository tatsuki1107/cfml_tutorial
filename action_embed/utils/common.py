from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


OPE_PALETTE = {
    "IPS": "tab:red",
    "DM": "tab:pink",
    "DR": "tab:blue",
    "MIPS (true)": "tab:gray",
    "MIPS (true) w/SLOPE": "tab:orange",
    "OFFCEM": "tab:green",
    "OFFCEM (1stage)": "tab:olive",
    "OFFCEM (cluster)": "tab:brown",
    "OFFCEM (LC)": "tab:purple",
}

OPL_PALLETE = {
    "IPS-PG": "tab:green",
    "MIPS-PG": "tab:purple",
    "DR-PG": "tab:red",
    "Reg-based": "tab:gray",
    "POTEC": "tab:orange",
    "OPL: POTEC, OPE: OFFCEM": "tab:blue",
    "POTEC (1stage)": "tab:olive",
    "POTEC (cluster)": "tab:brown",
}
TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 18
LINEWIDTH = 5
MARKERSIZE = 18


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


def visualize_mean_squared_error(
    result_df: DataFrame,
    xlabel: str,
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    plt.style.use("ggplot")
    
    estimators = result_df["estimator"].unique()
    palettes = {estimator: OPE_PALETTE[estimator] for estimator in estimators}
    xvalue = result_df["x"].unique()
    xvalue_labels = list(map(str, xvalue))

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    title = ["MSE", "Squared Bias", "Variance"]
    y = ["se", "bias", "variance"]

    ylims = []
    for i, (ax_, title_, y_) in enumerate(zip(axes, title, y)):
        sns.lineplot(
            data=result_df,
            x="x",
            y=y_,
            hue="estimator",
            style="estimator",
            ci=95 if i == 0 else None,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            ax=ax_,
            palette=palettes,
            markers=True,
            legend="full" if i == 0 else False,
        )
        if i == 0:
            handles, labels = ax_.get_legend_handles_labels()
            ax_.legend_.remove()
            for handle in handles:
                handle.set_linewidth(LINEWIDTH)
                handle.set_markersize(MARKERSIZE)

        # title
        ax_.set_title(f"{title_}", fontsize=TITLE_FONTSIZE)
        # xaxis
        if i == 1:
            ax_.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        else:
            ax_.set_xlabel("")

        ax_.set_xscale(xscale)
        ax_.set_xticks(xvalue, xvalue_labels, fontsize=TICK_FONTSIZE)
        ax_.get_xaxis().set_minor_formatter(plt.NullFormatter())

        # yaxis
        ax_.set_yscale(yscale)
        ax_.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax_.set_ylabel("")

        if i == 0:
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    if yscale == "linear":
        for ax_ in axes:
            ax_.set_ylim(ylims)

    fig.legend(
        handles,
        labels,
        fontsize=20,
        ncol=len(palettes),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def visualize_cdf_of_ralative_error(
    rel_result_df: DataFrame, baseline: str = "IPS"
) -> None:
    plt.style.use("ggplot")
    
    estimators = rel_result_df["estimator"].unique()
    palettes = {estimator: OPE_PALETTE[estimator] for estimator in estimators}
    baseline_se = rel_result_df[rel_result_df["estimator"] == baseline].set_index(
        "index"
    )["se"]
    rel_result_df["baseline_se"] = rel_result_df["index"].map(baseline_se)
    rel_result_df["rel_se"] = rel_result_df["se"] / rel_result_df["baseline_se"]
    rel_result_df = rel_result_df[["estimator", "rel_se"]]
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    sns.ecdfplot(
        linewidth=4,
        palette=palettes,
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


def visualize_learning_curve(curve_df: DataFrame) -> None:
    plt.style.use("ggplot")
    
    palletes = {k: OPL_PALLETE[k] for k in curve_df.method.unique()}
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    sns.lineplot(
        linewidth=5,
        x="index",
        y="rel_value",
        hue="method",
        style="method",
        ax=ax,
        palette=palletes,
        data=curve_df,
        legend="auto",
    )
    ax.set_title("Learning curve", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Rerative policy value", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("epochs", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)

    ax.axhline(1.0, color="black", linestyle="dotted", linewidth=3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        fontsize=TICK_FONTSIZE,
        ncol=len(palletes),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def visualize_test_value(result_df: DataFrame, x_label: str, x_scale: str) -> None:
    plt.style.use("ggplot")
    
    pallete = {k: OPL_PALLETE[k] for k in result_df.method.unique()}
    x_values = result_df.x.unique()
    
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    sns.lineplot(
        linewidth=5,
        markersize=15,
        markers=True,
        x="x",
        y="rel_value",
        hue="method",
        style="method",
        ax=ax,
        palette=pallete,
        data=result_df,
    )
    ax.set_title("Test policy value", fontsize=20)
    
    # xaxis
    ax.set_xscale(x_scale)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, fontsize=20)
    
    # legend
    ax.legend(fontsize=15, title="method", title_fontsize=15, loc="best")
    
    plt.show()
