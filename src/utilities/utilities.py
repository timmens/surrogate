"""Shared utility functions."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.data_management.train_test_split import NUM_TESTING_OBS_DICT
from src.specs import Specification


def load_data(
    model="kw_97_basic",
    testing=False,
    n_train=7000,
    n_test=None,
    sampling_suffix="random",
):
    """Return training and testing data.

    It is assumed that the outcome (quantity of interest) is stored in a column which
    starts with "qoi" and that none of the features share this property.

    Args:
        model (str): Model to choose the samples from, must be in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended']. Defaults to "kw_97_basic".
        testing (bool): Should the training data be used or testing. Defaults to False.
        n_train (int): Number of train observations to return. Defaults to 7000.
        n_test (int): Number of test observations to return. Defaults to the complete
            test set.
        sampling_suffix (str): Which sampling set to use. Defaults to "random".

    Returns:
        X, y (pd.DataFrame, pd.DataFrame): The features data frame and outcome series.

    """
    dataset = "test" if testing else "train"
    data_path = ppj("OUT_DATA", f"{dataset}-{model}-{sampling_suffix}")

    n_test = NUM_TESTING_OBS_DICT[model] if n_test is None else n_test

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


def load_model_specifications(model="kw_97_basic"):
    """Load specifcations for model fitting given (economic) ``model``.

    Args:
        model (str): Model type, must be in ["kw_94_one", "kw_97_basic",
            "kw_97_extended"]. Defaults to "kw_97_basic".

    Returns:
        specifcations (list): List of namedtuples of form
            ``src.specs.Specifcations``.

    """
    _ = Specification("id", "data_kwargs", "model_kwargs")  # flake8
    path = ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl")
    with open(path, "rb"):
        specifications = pickle.load(path)
    return specifications


def get_supported_surrogates(ignore=("__init__", "generic")):
    """Return list of currently implemented surrogate models.

    Load models implemented in directory ``src.surrogates`` and return model names in
    a list, except for models specified in ``ignore``.

    Args:
        ignore (tuple or list): Which files to ignore. Defaults to ('__init__',
            'generic').

    Returns:
        surrogates (list): List of names of supported models.

    """
    model_path = Path(ppj("IN_MODEL_CODE"))
    files = [f.stem for f in model_path.glob("*.py")]
    surrogates = [f for f in files if f not in ignore and not f.startswith(".")]
    return surrogates


def subset_features(X, order_features, ordered_features, n_features):
    """[summary]

    Args:
        X ([type]): [description]
        order_features ([type]): [description]
        ordered_features ([type]): [description]
        n_features ([type]): [description]

    Returns:
        [type]: [description]
    """
    if order_features and ordered_features is not None:
        feature_index = ordered_features[:n_features]
    else:
        np.random.seed(1)
        random_subset = np.random.choice(
            range(n_features), size=n_features, replace=False
        )
        feature_index = X.columns[random_subset]

    XX = X[feature_index].copy()
    return XX
