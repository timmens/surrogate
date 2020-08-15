import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bld.project_paths import project_paths_join as ppj


@click.command()
@click.argument("model", type=str)
def main(model):
    df = pd.read_csv(ppj("OUT_FINAL", f"{model}-losses_tidy.csv"))

    sns.set(font_scale=1.5, style="ticks")
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)  # A4 paper

    ax = sns.lineplot(x="n_obs", y="mae", hue="method", style="kwargs", data=df, ax=ax)
    ax.set_ylim(0, df["mae"].max())
    ax.set(
        xlabel="Number of obs. used for training", ylabel="Mean abs. error on test set"
    )
    ax.set(xscale="log")
    ax.set(title="Model performance given varying training data size")

    fig_path = ppj("OUT_FIGURES", f"{model}-mae_plot.pdf")
    fig.savefig(fig_path)


if __name__ == "__main__":
    main()  # pylint: disable-no-value
