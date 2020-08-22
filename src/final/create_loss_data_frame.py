from pathlib import Path

import click
import pandas as pd

from bld.project_paths import project_paths_join as ppj


def _kwarg_simplifier(kwargs):
    k = kwargs.replace("threshold-None", "None")
    return k


@click.command()
@click.argument("model", type=str)
def main(model):
    load_path = Path(ppj("OUT_ANALYSIS")) / model / "losses.csv"
    df = pd.read_csv(load_path)

    df = df.rename({"model": "method"}, axis=1)
    df["kwargs"] = df["kwargs"].apply(_kwarg_simplifier)
    df = df.set_index(["method", "n_obs", "kwargs"])

    save_path = Path(ppj("OUT_FINAL")) / f"{model}-losses_tidy.csv"
    df.to_csv(save_path)


if __name__ == "__main__":
    main()  # pylint: disable-no-value
