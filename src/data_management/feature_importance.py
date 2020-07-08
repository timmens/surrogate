from pathlib import Path

import click
import pandas as pd

from bld.project_paths import project_paths_join as ppj


@click.command()
@click.argument("model", type=str)
def main(model):
    file_name = Path(ppj("IN_DATA", f"{model}_indices.pkl"))
    if file_name.is_file():
        indices = pd.read_pickle(file_name)

        df = indices["mu_corr"].to_frame(name="mu_corr")
        df["name"] = ["_".join(tup) for tup in df.index]
        df.reset_index(inplace=True, drop=True)

        sorted_features = df.sort_values(by="mu_corr", ascending=False)["name"]
        sorted_features.to_csv(ppj("OUT_DATA", f"{model}-sorted_features.csv"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
