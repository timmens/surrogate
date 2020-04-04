import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    # the data
    with open(ppj("OUT_ANALYSIS", "variable_selection.pkl"), "rb") as handle:
        data = pickle.load(handle)

    df = data["df"]
    change_index = data["change_index"]
    changes = data["changes"]
    thresholds = data["thresholds"]

    # the plot
    sns.set(style="ticks", rc={"lines.linewidth": 3}, font_scale=1.2)
    fig, ax = plt.subplots()

    fig.set_size_inches(13, 8.27)
    ax.set(xlim=(0, 0.023))
    ax.set(ylim=(0.005, 0.025))

    np.random.seed(4)
    sns.lineplot(x="thresholds", y="mae", data=df)
    i = 0
    for change, change_text in zip(change_index, changes):
        i += 1
        if i > 15:
            break
        add = -0.001 * i
        change_thresh = thresholds[change]
        plt.axvline(x=change_thresh, color="r", linewidth=0.5)
        plt.annotate(xy=(change_thresh, 0.02 + add), s=change_text, weight="bold")
    plt.title("Variable selection using Ridge Regression")
    plt.xlabel("Coefficient threshold")
    plt.ylabel("Mean abs. error on test set")
    plt.annotate(
        xy=(0.0005, 0.0205),
        s="Blue line shows the mean absolute error when using\na "
        "newly fit 2nd degree polynomial that only\nuses "
        "variables which had a coefficient value\nabove the "
        "threshold. Annotated red lines display\nwhen and which "
        "variable was dropped. (Variables\n included were"
        "standardized)",
        bbox={"facecolor": "white", "edgecolor": "gray", "boxstyle": "round"},
    )

    fig_path = ppj("OUT_FIGURES", "ridge_variable_selection.pdf")
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.7)
