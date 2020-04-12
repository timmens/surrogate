import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    df = pd.read_csv(ppj("OUT_FINAL", "losses_mae_tidy.csv"))

    df = df.query("features==27")

    sns.set(font_scale=1.5, style="ticks")

    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)  # A4 paper

    ax = sns.lineplot(x="samples", y="mae", hue="method", data=df, ax=ax)
    ax.set_ylim(0.004, 0.025)
    ax.set(
        xlabel="Number of obs. used for training", ylabel="Mean abs. error on test set"
    )
    ax.set(xscale="log")
    ax.set(title="Comparison of model performance for varying training sample size")

    fig_path = ppj("OUT_FIGURES", "mae_plot.pdf")
    fig.savefig(fig_path)
