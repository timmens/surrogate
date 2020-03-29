import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    df = pd.read_csv(ppj("OUT_ANALYSIS", "bootstrap_mae.csv"))

    sns.set(font_scale=1.5, style="ticks")

    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)  # A4 paper

    ax = sns.lineplot(x="n", y="mae", hue="model", data=df, ci=99, ax=ax)
    ax.set_ylim(0.005, 0.015)
    ax.set(
        xlabel="Number of obs. used for training", ylabel="Mean abs. error on test set"
    )
    ax.set(xscale="log")
    ax.set(title="Comparison of model performance for varying training sample size")
    ax.text(
        95,
        0.0055,
        "Bootstrap draws: 50",
        bbox={"facecolor": "none", "edgecolor": "gray", "boxstyle": "round"},
    )

    fig_path = ppj("OUT_FIGURES", "bootstrap_mae.pdf")
    fig.savefig(fig_path)

    # move figure to sciebo
    sciebo_path = "/home/tm/sciebo/uni-master/master-thesis/structUncertainty/"
    shutil.copy(fig_path, sciebo_path + "bootstrap_mae.pdf")
