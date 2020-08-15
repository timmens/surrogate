import click
import pandas as pd

from bld.project_paths import project_paths_join as ppj


NUM_TESTING_OBS_DICT = {
    "kw_94_one": 30000,
    "kw_97_basic": 3000,
    "kw_97_extended": 6930,
}


@click.command()
@click.argument("model", type=str)
def main(model):
    # split samples into training and testing set
    file_name = ppj("IN_DATA", f"samples-{model}.pkl")
    sample = pd.read_pickle(file_name)

    n_obs_testing = NUM_TESTING_OBS_DICT[model]
    testing = sample.iloc[:n_obs_testing]
    training = sample.iloc[n_obs_testing:]

    # add multiindex and save
    for df, name in zip([testing, training], ["test", "train"]):
        df.reset_index(drop=True, inplace=True)
        df.rename_axis("iteration", inplace=True)
        df.insert(0, "dataset", name)
        df.set_index("dataset", append=True, inplace=True)
        df.reorder_levels(["dataset", "iteration"])
        df.to_pickle(ppj("OUT_DATA", f"{name}-{model}.pkl"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
