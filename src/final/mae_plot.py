from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bld.project_paths import project_paths_join as ppj


def translations():
    kwarg_translation = {
        "f-degree-1-p-None": "Linear",
        "f-degree-2-p-None": "Quadratic",
        "f-layers-88-88-88-88-88-n-epochs-200-n-batch-size-128-p-None": "Large",
        "f-layers-88-88-88-n-epochs-200-n-batch-size-128-p-None": "Medium",
        "f-layers-88-88-n-epochs-200-n-batch-size-128-p-None": "Small",
        "f-learning-rate-005-p-None": "SlowLearner",
        "f-learning-rate-01-p-None": "FastLearner",
    }
    return kwarg_translation


@click.command()
@click.argument("model", type=str)
def main(model):
    load_path = Path(ppj("OUT_FINAL")) / f"{model}-losses_tidy.csv"
    df = pd.read_csv(load_path)

    kwarg_translations = translations()
    df = df.replace(kwarg_translations)

    methods = list(set(df.method))
    sns.set(font_scale=1.5, style="ticks")

    fig, axs = plt.subplots(1, len(methods), sharex=True, sharey=True)
    fig.set_size_inches(11.7, 8.27)  # A4 paper
    fig.suptitle("Model performance given varying training data size")

    for ax, method in zip(axs, methods):
        ax = sns.lineplot(
            x="n_obs", y="mae", hue="kwargs", data=df.query("method==@method"), ax=ax
        )
        ax.set(xscale="log")
        ax.set_ylim(0, df["mae"].max())
        ax.set_xlim(200, 22000)
        ax.legend(title=method, loc="upper center", bbox_to_anchor=(0.5, 0.3))

    fig_path = Path(ppj("OUT_FIGURES")) / f"{model}-mae_plot.png"
    fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    main()  # pylint: disable-no-value
