import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.specs import read_specifications


NUM_TESTING_OBS_DICT = {
    "kw_94_one": 30000,
    "kw_97_basic": 3000,
    "kw_97_extended": 8616,
}


def split_data(data, data_set):
    """Split data set and return sets in dictionary."""
    n_obs_testing = NUM_TESTING_OBS_DICT[data_set]
    testing = data.iloc[:n_obs_testing]
    training = data.iloc[n_obs_testing:]
    splitted_data = (testing.reset_index(drop=True), training.reset_index(drop=True))
    return splitted_data


def save_data(splitted_data, produces):
    """Save splitted data sets."""
    for target, data_set in zip(produces, splitted_data):
        data_set.to_pickle(target)


def load_specifications():
    """Load specifications and return as list of lists."""
    data_sets = load_data_set_names()
    produces = [
        (BLD / "data" / f"test-{data_set}.pkl", BLD / "data" / f"train-{data_set}.pkl",)
        for data_set in data_sets
    ]
    return zip(produces, data_sets)


def load_data_set_names():
    specifications = read_specifications(fitting=True)
    data_sets = set()
    for _, spec in specifications.items():
        data_sets = data_sets.union(set(spec["data_set"]))

    data_sets = list(data_sets)
    return data_sets


def load_data(name):
    """Read data set given name."""
    data = pd.read_pickle(SRC / "data" / f"samples-{name}.pkl").rename(
        {name: "iteration"}
    )
    return data


@pytask.mark.parametrize("produces, data_set", load_specifications())
def task_train_test_split(produces, data_set):
    data = load_data(data_set)
    splitted_data = split_data(data, data_set)
    save_data(splitted_data, produces)
