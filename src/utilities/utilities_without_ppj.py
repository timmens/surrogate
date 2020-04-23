import importlib
import os
import sys
from contextlib import contextmanager


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


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
