"""Task file which splits data sets into training and testing sets.

Number of testing observations is specified in the below dictionary, where the key is
the name of the data set. All other observations are then used to construct the
training data set.

Important: In a generic specification, stored in ``src/specs``, the key "data_set"
must coincide with the file name stored in ``src/data``, which has to be of the form
"samples-'data_set'.pkl". For example "samples-kw_94_one.pkl" coincides with a key set
as "data_set: kw_94_one" in a specification. If a new data set is added to the data
folder then a new entry of number of test observations has to be added to the below
dictionary "NUM_TESTING_OBS_DICT".

"""
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


def load_data(name):
    """Read data set given name."""
    data = pd.read_pickle(SRC / "data" / f"samples-{name}.pkl").rename(
        {name: "iteration"}
    )
    return data


def split_data(data, data_set):
    """Split data set and return sets in dictionary."""
    n_obs_testing = NUM_TESTING_OBS_DICT[data_set]
    testing = data.iloc[:n_obs_testing]
    training = data.iloc[n_obs_testing:]
    splitted_data = (testing.reset_index(drop=True), training.reset_index(drop=True))
    return splitted_data


def save_data(splitted_data, produces):
    """Save splitted data sets."""
    produces = list(produces.values())
    for target, data_set in zip(produces, splitted_data):
        data_set.to_pickle(target)


def load_specifications():
    """Load specifications and return as list of lists (for pytask)."""
    data_sets = load_data_set_names()
    produces = [
        (BLD / "data" / f"test-{data_set}.pkl", BLD / "data" / f"train-{data_set}.pkl")
        for data_set in data_sets
    ]
    return zip(produces, data_sets)


def load_data_set_names():
    """Load data set names which are needed for given specifications."""
    specifications = read_specifications(fitting=True)
    data_sets = set()
    for _, spec in specifications.items():
        data_sets = data_sets.union(set(spec["data_set"]))

    data_sets = list(data_sets)
    return data_sets


@pytask.mark.parametrize("produces, data_set", load_specifications())
def task_train_test_split(produces, data_set):
    data = load_data(data_set)
    splitted_data = split_data(data, data_set)
    save_data(splitted_data, produces)
