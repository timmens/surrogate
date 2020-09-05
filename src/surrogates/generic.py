"""Generic surrogate module.

Allows for the fitting, prediction, saving and loading of implemented surrogate models.

Example:

```
    import src.surrogate as surrogate
    nnet = surrogate.fit("neuralnet", X, y)
    pred = surrogate.predict(nnet, XX)
    surrogate.save(nnet, "fitted-nnet")
    nnet = surrogate.load("fitted-nnet")
```

"""
import warnings
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import tensorflow as tf

from src.surrogates import catboost
from src.surrogates import neuralnetwork
from src.surrogates import polynomialregression
from src.surrogates import ridgeregression


def fit(model_type, X, y, **kwargs):
    """Fit and return a surrogate model.

    Args:
        model_type (str): Surrogate model type. Must be in ['polynomial', 'ridge',
            'neuralnet'].
        X (pd.DataFrame, pd.Series or np.ndarray): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes

    Returns:
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline. 'pipe' is a
            sklearn.pipeline.Pipeline object. 'model' depends on the module used.

    """
    assert len(X) == len(y)
    assert isinstance(X, (np.ndarray, pd.DataFrame, pd.Series))
    assert isinstance(y, (np.ndarray, pd.DataFrame, pd.Series))

    module = _model_name_to_module(model_type)
    predictor = module.fit(X, y, **kwargs)
    predictor["model_type"] = model_type
    return predictor


def predict(X, predictor, **kwargs):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame, pd.Series or np.ndarray): New data on features. Must be
            comformable with the data used in the fitting process.
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline. 'pipe' is a
            sklearn.pipeline.Pipeline object. 'model' depends on the module used.

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    assert isinstance(X, (np.ndarray, pd.DataFrame, pd.Series))
    module = _model_name_to_module(predictor["model_type"])
    predictions = module.predict(X, predictor, **kwargs)
    return predictions


def save(predictor, file_path="", overwrite=False):
    """Save fitted model to disc.

    Args:
        predictor (dict): Dictionary with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline. 'pipe' is a
            sklearn.pipeline.Pipeline object. 'model' depends on the module used.
        file_path (str): File path. Defaults to active path of session.
        overwrite (bool): Should the file be overwritten if it exists. Default is False.

    Returns:
        None

    """
    f = Path(file_path)
    file_path = f if f.suffix == ".pkl" else f.with_suffix(".pkl")

    if "neuralnet" in file_path.stem:
        predictor["model"].model.save(file_path.with_suffix(".nnet_model"))
        predictor["model"] = None

    if not file_path.is_file() or overwrite:
        with open(file_path, "wb") as f:
            cloudpickle.dump(predictor, f)
    else:
        warnings.warn("File already exists. No actions taken.", UserWarning)


def load(file_path):
    """Load a fitted model from disc.

    Args:
        file_path (str): File path.

    Returns:
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline. 'pipe' is a
            sklearn.pipeline.Pipeline object. 'model' depends on the module used.

    """
    f = Path(file_path)
    file_path = f if f.suffix == ".pkl" else f.with_suffix(".pkl")
    with open(file_path, "rb") as f:
        predictor = cloudpickle.load(f)

    if "neuralnet" in file_path.stem:
        model = tf.keras.models.load_model(file_path.with_suffix(".nnet_model"))
        predictor["model"] = model

    return predictor


def _model_name_to_module(model_type):
    """Translate name of predictors to module names.

    Args:
        model_type (str): Type name of surrogate predictors.

    Returns:
        translated (str): Name of module corresponding to name of predictor.

    Example:
    >>> import src.surrogate as surrogate
    >>> lm = surrogate.fit("polynomial", X, y, degree=2)
    >>> model_name = type(lm).__name__
    >>> _model_name_to_module_name(model_name)
    "polynomial"

    """
    modules = {
        "PolynomialRegressor": polynomialregression,
        "RidgeRegressor": ridgeregression,
        "NeuralnetRegressor": neuralnetwork,
        "CatBoostRegressor": catboost,
    }
    try:
        module = modules[model_type]
    except KeyError:
        print(f"{model_type} regressor not implemented.")
        raise KeyError
    return module
