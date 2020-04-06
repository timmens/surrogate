"""Polynomial regression surrogate model."""
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from src.model_code.surrogate import assert_input_fit
from src.model_code.surrogate import Surrogate
from src.utilities.utilities import get_feature_names


class PolynomialRegression(Surrogate):
    """Polynomial regression surrogate model."""

    def __init__(self):
        self.degree = None
        self.coefficients = None
        self.scaler = None
        self.is_fitted = False
        self.file_type = ".csv"
        super().__init__()

    def fit(self, X, y, **kwargs):
        """Fit a polynomial regression model using least squares.

        Args:
            X (pd.DataFrame):
                Data on features.

            y (pd.Series or np.ndarray):
                Data on outcomes.

            **kwargs:
                - 'degree' (int): Degree of the polynomial model.
                - 'fit_intercept' (bool): Should an intercept be fitted.
                - 'n_jobs' (int): Number of jobs to use for parallelization.

        Returns:
            self: The fitted PolynomailRegression object.

        """
        model = _fit(X=X, y=y, **kwargs)

        self.degree = kwargs["degree"]
        self.coefficients = model["coefficients"]
        self.scaler = model["scaler"]
        self.is_fitted = True

        return self

    def predict(self, X):
        """Predict outcome using the fitted model and new data.

        Uses the fitted coefficients in ``self.coefficients`` and new data
        ``X`` to predict outcomes. If the model has not been fitted yet we
        return None and a warning.

        Args:
            X (pd.DataFrame): New data on features (unprocessed).

        Returns:
            - None, if ``self.is_fitted`` is False and otherwise
            - predictions (np.array): The predicted outcomes.

        """
        if not self.is_fitted:
            warnings.warn("The model has not been fitted yet.", UserWarning)
            return None

        predictions = _predict(
            X=X, coefficients=self.coefficients, scaler=self.scaler, degree=self.degree
        )
        return predictions

    def save(self, filename, overwrite=False):
        """Save fitted model to disc.

        Save the fitted coefficents stored under ``self.coefficients`` to disc
        as a csv file. Can be loaded afterwards using the method ``load``.

        Args:
            filename (str): File path except file type ending. If file type ending is
                provided it will be ignored.
            overwrite (bool): Should the file be overwritten if it exists.

        Returns:
            None

        """
        file_path, file_type = os.path.splitext(filename)
        _save(
            coefficients=self.coefficients,
            degree=self.degree,
            scaler=self.scaler,
            file_path=file_path,
            format=self.file_type,
            overwrite=overwrite,
        )

    def load(self, filename):
        """Load a fitted model from disc.

        Load fitted coefficients stored as a csv from disc and store them under
        ``self.coefficients``. Will work on files created by method ``save``,
        but also files manually created with first named column "coefficient"
        and second column "value".

        Args:
            filename (str): File path except file type ending. If file type ending is
                provided it will be ignored.

        Returns:
            None

        """
        file_path, file_type = os.path.splitext(filename)
        model = _load(file_path, self.file_type)

        self.coefficients = model["coefficients"]
        self.scaler = model["scaler"]
        self.degree = model["degree"]
        self.is_fitted = True

        return self

    @property
    def name(self):
        """Return name of class as string."""
        return self.__class__.__name__


def _fit(X, y, degree, fit_intercept=False, n_jobs=1):
    """Fit a polynomial regression model using least squares.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        degree (int): Degree of the polynomial model.
        fit_intercept (bool): Should an intercept be fitted.
        'n_jobs' (int): Number of jobs to use for parallelization.

    Returns:
        out (dict): Output dictionary containing:
            - coefficients (pd.DataFrame): The named coefficient values.
            - scaler (sklearn.preprocessing.StandardScaler): Fitted scaler.

    """
    assert_input_fit(X=X, y=y)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_ = poly.fit_transform(X)

    scaler = StandardScaler()
    scaler = scaler.fit(X_)
    XX_ = scaler.transform(X_)

    lm = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
    lm = lm.fit(X=XX_, y=y)

    coef_values = lm.coef_.reshape((-1,))
    coef_names = get_feature_names(poly, X)
    if fit_intercept:
        coef_values = np.insert(coef_values, 0, lm.intercept_)
        coef_names = ["intercept"] + coef_names

    coefficients = pd.DataFrame(
        data=zip(coef_names, coef_values), columns=["coefficient", "value"]
    ).set_index("coefficient")

    out = {
        "coefficients": coefficients,
        "scaler": scaler,
    }
    return out


def _predict(X, scaler, coefficients, degree):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used to fit.
        coefficients (pd.DataFrame): The named coefficient values.
        degree (int): Degree of the polynomial model.


    Returns:
        predictions (np.array): The predicted outcomes.

    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_ = poly.fit_transform(X)
    XX_ = scaler.transform(X_)

    has_intercept = coefficients.index[0] == "intercept"
    if has_intercept:
        intercept = coefficients.loc["intercept", "value"]
        coef = coefficients.drop("intercept")
    else:
        coef = coefficients

    coef = coef.reindex(get_feature_names(poly=poly, X=X))  # sort coefficients

    predictions = XX_.dot(coef)
    predictions = predictions.reshape((-1,))
    predictions = predictions + intercept if has_intercept else predictions

    return predictions


def _save(coefficients, degree, scaler, file_path, format, overwrite):
    """Save fitted model to disc.

    Save the fitted coefficents to disc as a csv file. Since the number of degrees of
    the polynomial is important we create a new row with name "<--degree-->" and value
    ``degree``. We choose the weird name so that the chance is low that a standard
    feature name is equal to it.

    Args:
        coefficients (pd.DataFrame): The named coefficient values.
        degree (int): Degree of the polynomial model.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used to fit.
        file_parth (str): File path.
        format (str): File format of saved model coefficients.
        overwrite (bool): Should the file be overwritten if it exists.

    Returns:
        None

    """
    out = coefficients.copy()
    out.loc["<--degree-->"] = degree

    file_present = glob.glob(file_path + format)
    if overwrite or not file_present:
        out.to_csv(file_path + format)
        with open(file_path + "_scaler.pkl", "wb") as handle:
            pickle.dump(scaler, handle)
    else:
        warnings.warn("File already exists. No actions taken.", UserWarning)


def _load(file_path, format):
    """Load a fitted model from disc.

     Load fitted coefficients stored as a csv from disc and return them. Will
     work on files created by method ``save``, but also files manually created
     with first named column "coefficient", which will be set as the index.
     For correct functionality of the other functions the second column needs
     to be called "value". If the file to load was not created using the ``save``
     method one has to manually add a row with name "<--degree-->" and value equal to
     the degree of the polynomial which correspond to the coefficients.

    Args:
        file_path (str): File path.
        format (str): Format of how model is saved. Example: format=".csv".

    Returns:
        coefficients (pd.DataFrame): The named coefficient values.

    Raises:
        ValueError, if index column name is wrongly specified in csv file.
        AssertionError, if value column name is incorrect.

    """
    try:
        coefficients = pd.read_csv(file_path + format, index_col="coefficient")
        assert set(coefficients.columns) == {"value"}
        degree = int(coefficients.loc["<--degree-->"])
        coefficients = coefficients.drop("<--degree-->")

        with open(file_path + "_scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)
    except ValueError:
        raise ValueError(
            "Columns have wrong names. One column has to be"
            "called 'coefficient' containing coefficient names,"
            "and one column has to be called 'value', containing"
            "coefficient values."
        )

    out = {
        "coefficients": coefficients,
        "scaler": scaler,
        "degree": degree,
    }
    return out
