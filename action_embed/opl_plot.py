import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

PALLETE = {
    "logging": "tab:grey",
    "IPS-PG": "tab:red",
    "MIPS-PG": "tab:green",
}


def visualize_learning_curve(curve_df: pd.DataFrame, pi_b_value: np.float64) -> None:
    
    pallete = {k: PALLETE[k] for k in curve_df.method.unique()}
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    
    curve_df["rel_value"] = curve_df.value / pi_b_value
    sns.lineplot(
        linewidth=5,
        x="index",
        y="rel_value",
        hue="method",
        style="method",
        ax=ax,
        palette=pallete,
        data=curve_df,
    )
    ax.set_title("Learning curve", fontsize=20)
    ax.set_ylabel("Rerative policy value", fontsize=30)
    ax.set_xlabel("epochs", fontsize=25)
    ax.tick_params(axis="y", labelsize=20)
    
    ax.legend(fontsize=15, title="method", title_fontsize=15, loc='lower right')
    plt.show()
    
    
def visualize_test_value(result_df: pd.DataFrame, x_label: str, x_scale: str) -> None:
    pallete = {k: PALLETE[k] for k in result_df.method.unique()}
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
