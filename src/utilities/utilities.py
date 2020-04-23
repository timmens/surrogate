"""Helper functions."""
import json
import pathlib
import pickle
import re
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


def extract_features_from_data(df):
    """Extract features from data frame.

    Args:
        df (pd.DataFrame): A data frame with columns as in the original df.

    Returns:
        X (pd.DataFrame): A data frame containing only features.

    """
    feature_column = ~df.columns.str.startswith("qoi")
    X = df.loc[:, feature_column].copy()
    return X


def extract_outcome_from_data(df, outcome=1):
    """Extract specific utcome from data frame.

    Args:
        df (pd.DataFrame): A data frame with columns as in the original df.
        outcome (int): Which outcome to select. Possible values: 1, 2, 3.

    Returns:
        y (np.ndarray): The specified outcome.

    """
    outcome_column = "qoi_tuition_subsidy_" + str(500 * outcome)
    return df[outcome_column].values


def load_testing_data(nobs=25000):
    """Load data for testing.

    Args:
        nobs (int): The number of testing observations.
            Has to be in the range [0, 25000].

    Returns:
        X, y (np.array): Testing features and testing outcomes.

    """
    df = pickle.load(open(ppj("OUT_DATA", "df_validation.pkl"), "rb"))

    df = df.iloc[:nobs, :]

    Xtest = extract_features_from_data(df)
    ytest = extract_outcome_from_data(df)

    return Xtest, ytest


def load_training_data(nobs=75000, seed=1):
    """Load data for testing and training.

    Args:
        nobs (int): The number of training observations.
            Has to be in the range [0, 75000].
        seed (int): Random number seed.

    Returns:
        X, y (np.array): Training features and Training outcomes.
    """
    df = pickle.load(open(ppj("OUT_DATA", "df_training.pkl"), "rb"))

    np.random.seed(seed)
    index = np.random.choice(np.arange(75000), size=nobs, replace=False)

    df = df.iloc[index, :]

    X = extract_features_from_data(df)
    y = extract_outcome_from_data(df)

    return X, y


def get_feature_names(poly, X):
    """Extract feature names of polynomial features.

    Args:
        poly: fitted ``sklearn.preprocessing.PolynomialFeatures`` object.
        X (pd.DataFrame): Data on features.

    Returns:
        coef_names (list): List of feature names corresponding to all polynomials.
    """
    coef_names = poly.get_feature_names(X.columns)
    coef_names = [name.replace(" ", ":") for name in coef_names]
    return coef_names


def compute_testing_loss(model, ytest, Xtest, measure, **kwargs):
    ypred = model.predict(Xtest, **kwargs)
    mae = measure(ytest, ypred)
    return mae


def compute_loss_given_metrics(ytrue, ypredicted, metrics):
    """Compute loss of prediction given various metrics.

    Args:
        ytrue (np.array): True outcome.
        ypredicted (np.array): Predicted outcome.
        metrics (dict): Dictionary containing metrics.

    Returns:
        loss (pd.Series): The occured loss from prediction.

    """
    loss = []
    for _, metric in metrics.items():
        loss.append(metric(ytrue, ypredicted))

    loss = pd.Series(loss, index=metrics.keys())
    return loss


def load_implemented_models(ignore=("__init__", "surrogate")):
    """Load implemented models.

    Load models implemented in directory ``src.model_code`` and return model names as
    a list while not return models specified in the argument *ignore*.

    Args:
        ignore (tuple or list): Which files to ignore. Defaults to ('__init__',
            'surrogate').

    Returns:
        files (list): List of names of implemented modelsc

    """
    p = pathlib.Path(ppj("IN_MODEL_CODE", ""))
    files = [str(file) for file in p.glob("*.py")]

    if len(files) == 0:
        warnings.warn("No models implemented.", UserWarning)
        return None

    files = [_extract_filename_from_path(file) for file in files]
    files = [file for file in files if file not in ignore]
    return files


def _extract_filename_from_path(file):
    """Extract name of file from complete directory path.

    Args:
        file (str): File path.

    Returns:
        f (str): The name of the file.
    """
    f = file.split("/")[-1]
    f = f.split(".")[0]
    return f


def load_surrogates_specs():
    """Load model specifications for ``fit`` method of surrogates.

    Returns:
        out (dict): Dictionary containing parameters for ``fit`` method of surrogates.

    """
    filepath = ppj("OUT_MODEL_SPECS", "model_specs.json")

    with open(filepath) as file:
        out = json.loads(file.read())
    return out


def load_sorted_features():
    """Load features sorted with resepect to importance indices.

    Returns:
        out (pd.Series): Sorted series containing features.

    """
    series = pd.read_csv(
        ppj("OUT_DATA", "sorted_features.csv"), header=None, names=["name"]
    )
    return series.name


def load_implemented_model_names():
    models = load_model_information()
    names = {model.name for model in models}
    return list(names)


def load_model_information():
    """Load model information as namedtuple.

    Returns:
        models (list): List of namedtuples which represent the implemented
          models. An item from `models` has attribured `colname` for an individual
          string key, `name` for the name of the model, `label` for a label which
          can be printed on a plot, `n` for the number of observations used for
          fitting the model and `p` for the number of features used for fitting
          of the model. Example.
          models[0] = Model(colname='nnet1_p27_n50000', name='nnet1', label=
          'Neural Network', n=50000, p=27)
    """
    model_specs = load_surrogates_specs()

    # extract unique model names
    model_keys = list(model_specs.keys())
    model_names = [model.split("_")[0] for model in model_keys]

    # extract number of observation grid and number of features grid
    data_info = [model.split("_")[1:] for model in model_keys]
    nobs = _select_n(data_info)

    nfeatures = _select_p(data_info)

    # store information about each specification in a namedtuple
    Model = namedtuple("Model", "colname name n p")
    models = [
        Model(colname=key, name=name, n=n, p=p)
        for key, name, n, p in zip(model_keys, model_names, nobs, nfeatures)
    ]
    return models


def _select_n(data_info):
    """Select number of observations from list of list of model data information.

    Args:
        data_info (list): List of list of model information.

    Returns:
        nobs (list): List containing number of observations for each item in data_info.

    """
    obs_data = [e for info in data_info for e in info if str.startswith(e, "n")]
    nobs = [int(s) for e in obs_data for s in re.findall(r"\d+", e)]
    return nobs


def _select_p(data_info):
    """Select number of features from list of list of model data information.

    Args:
        data_info (list): List of list of model information.

    Returns:
        nfeatures (list): List containing number of features for each item in data_info.

    """
    feature_data = [e for info in data_info for e in info if str.startswith(e, "p")]
    nfeatures = [int(s) for e in feature_data for s in re.findall(r"\d+", e)]
    return nfeatures
