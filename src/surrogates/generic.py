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
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.surrogates import polynomialregression
from src.surrogates import ridgeregression


MODEL_MODULES = {
    "polynomial": polynomialregression,
    "ridge": ridgeregression,
}


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
    assert model_type in MODEL_MODULES.keys()

    module = MODEL_MODULES[model_type]
    predictor = module.fit(X, y, **kwargs)
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

    model_type = _model_name_to_module_name(type(predictor).__name__)
    module = MODEL_MODULES[model_type]
    predictions = module.predict(X, predictor, **kwargs)
    return predictions


def save(predictor, file_path="", overwrite=False):
    """Save fitted model to disc.

    Args:
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline. 'pipe' is a
            sklearn.pipeline.Pipeline object. 'model' depends on the module used.
        file_path (str): File path. Defaults to active path of session.
        overwrite (bool): Should the file be overwritten if it exists. Default is False.

    Returns:
        None

    """
    f = Path(file_path)
    file_path = f if f.suffix == ".pkl" else f.with_suffix(".pkl")

    if not file_path.is_file() or overwrite:
        with open(file_path, "wb") as f:
            pickle.dump(predictor, f)
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
        predictor = pickle.load(f)
    return predictor


def _model_name_to_module_name(model_name):
    """Translate name of predictors to module names.

    Args:
        model_name (str): Type name of surrogate predictors.

    Returns:
        translated (str): Name of module corresponding to name of predictor.

    Example:
    >>> import src.surrogate as surrogate
    >>> lm = surrogate.fit("polynomial", X, y, degree=2)
    >>> model_name = type(lm).__name__
    >>> _model_name_to_module_name(model_name)
    "polynomial"

    """
    translation = {
        "PolynomialPredictor": "polynomial",
        "RidgePredictor": "ridge",
        "NeuralnetPredictor": "neuralnet",
    }
    translated = translation[model_name]
    return translated
