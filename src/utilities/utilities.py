"""Helper functions."""
import importlib
import json
import os
import pathlib
import pickle
import sys
import warnings
from contextlib import contextmanager

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
    df = pickle.load(open(ppj("OUT_DATA", "data_test.pkl"), "rb"))

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
    df = pickle.load(open(ppj("OUT_DATA", "data_train.pkl"), "rb"))

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


def get_model_class_name(model):
    """Translate model names to names of respective classes.

    Args:
        model (str): Model name.

    Returns:
        translated (list): List model class names.
    """
    translation = {
        "linearregression": "LinearRegression",
        "polynomialregression": "PolynomialRegression",
        "ridgeregression": "RidgeRegression",
        "neuralnetwork": "NeuralNetwork",
    }

    return translation[model]


def get_model_class_names(models):
    """

    Args:
        models:

    Returns:

    """
    return [get_model_class_name(model) for model in models]


def get_surrogate_instance(surrogate):
    """

    Args:
        surrogate:

    Returns:

    """
    module = "src.model_code." + surrogate
    module = importlib.import_module(module)
    surrogate_class_name = get_model_class_name(surrogate)
    surrogate_class = getattr(module, surrogate_class_name)()
    return surrogate_class


def get_surrogate_instances(surrogates):
    """

    Args:
        surrogates:

    Returns:

    """
    if isinstance(surrogates, list) or isinstance(surrogates, tuple):
        out = [get_surrogate_instance(surrogate) for surrogate in surrogates]
    else:
        out = get_surrogate_instance(surrogates)

    return out


def load_surrogates_specs():
    """Load model specifications for ``fit`` method of surrogates.

    Returns:
        out (dict): Dictionary containing parameters for ``fit`` method of surrogates.

    """
    with open(ppj("OUT_MODEL_SPECS", "model_specs.json")) as file:
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


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
