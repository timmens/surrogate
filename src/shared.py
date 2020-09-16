"""Shared utility functions."""
import pandas as pd
import yaml

from src.config import BLD
from src.config import SRC
from src.task_train_test_split import NUM_TESTING_OBS_DICT


def load_data(
    data_set="kw_97_basic", testing=False, n_train=7000, n_test=None,
):
    """Return training and testing data.

    It is assumed that the outcome (quantity of interest) is stored in a column which
    starts with "qoi" and that none of the features share this property.

    Args:
        data_set (str): Model to choose the samples from, must be in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended']. Defaults to "kw_97_basic".
        testing (bool): Should the training data be used or testing. Defaults to False.
        n_train (int): Number of train observations to return. Defaults to 7000.
        n_test (int): Number of test observations to return. Defaults to the complete
            test set.
        sampling_suffix (str): Which sampling set to use. Defaults to "random".

    Returns:
        X, y (pd.DataFrame, pd.DataFrame): The features data frame and outcome series.

    """
    data_set_type = "test" if testing else "train"
    data_path = BLD / "data" / f"{data_set_type}-{data_set}.pkl"

    n_test = NUM_TESTING_OBS_DICT[data_set] if n_test is None else n_test
    n_obs = n_test if testing else n_train

    df = pd.read_pickle(data_path)
    assert (
        0 <= n_obs <= len(df)
    ), "Number of observations cannot be greater than actual data set."

    df = df.iloc[:n_obs, :]

    outcome = df.columns[df.columns.str.startswith("qoi")]
    y = df.loc[:, outcome]
    X = df.drop(outcome, axis=1)
    return X, y


def get_model_and_kwargs_from_type(surrogate_type):
    """Load kwargs given model specified at ``src/surrogates/name_to_kwargs.yaml``."""
    path = SRC / "surrogates" / "name_to_kwargs.yaml"
    name_to_kwargs = yaml.safe_load(open(path, "r"))
    model, kwargs = tuple(name_to_kwargs[surrogate_type].values())
    return model, kwargs
