import click
import pandas as pd

from bld.project_paths import project_paths_join as ppj


def _kwarg_simplifier(kwargs):
    k = kwargs.replace("threshold-None", "None")
    return k


@click.command()
@click.argument("model", type=str)
def main(model):
    df = pd.read_csv(ppj("OUT_ANALYSIS", f"{model}-losses.csv"))
    df = df.rename({"model": "method"}, axis=1)
    df["kwargs"] = df["kwargs"].apply(_kwarg_simplifier)
    df = df.set_index(["method", "n_obs", "kwargs"])
    df.to_csv(ppj("OUT_FINAL", f"{model}-losses_tidy.csv"))


if __name__ == "__main__":
    main()  # pylint: disable-no-value
