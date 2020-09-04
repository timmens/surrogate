import pandas as pd

from src.config import BLD
from src.config import SRC
from src.specs import read_specifications


NUM_TESTING_OBS_DICT = {
    "kw_94_one": 30000,
    "kw_97_basic": 3000,
    "kw_97_extended": 8616,
}


def split_data(data_sets):
    """Split data sets and return in big dictionary."""
    data = {}
    for name, df in data_sets.items():
        n_obs_testing = NUM_TESTING_OBS_DICT[name]
        testing = df.iloc[:n_obs_testing]
        training = df.iloc[n_obs_testing:]
        data[f"test-{name}"] = testing.reset_index(drop=True)
        data[f"train-{name}"] = training.reset_index(drop=True)
    return data


def save_data(data):
    """Save data."""
    p = BLD / "data"
    p.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_pickle((p / name).with_suffix(".pkl"))


def get_data_set_names(specifications):
    """Get list of unique data sets that need to be split."""
    data_sets = {spec["data_set"] for spec in specifications.values()}
    return list(data_sets)


def read_data(names):
    """Read data sets given names."""
    data_sets = {
        name: pd.read_pickle(SRC / "data" / f"samples-{name}.pkl").rename_axis(
            "iteration"
        )
        for name in names
    }
    return data_sets


def main():
    specifications = read_specifications()
    data_set_names = get_data_set_names(specifications)
    data_set_dict = read_data(data_set_names)
    splitted_set_dict = split_data(data_set_dict)
    save_data(splitted_set_dict)


if __name__ == "__main__":
    main()
