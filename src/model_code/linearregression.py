"""Linear regression surrogate model."""
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LinReg

from src.model_code.surrogate import assert_input_fit
from src.model_code.surrogate import Surrogate


class LinearRegression(Surrogate):
    """Linear regression surrogate model."""

    def __init__(self):
        self.coefficients = None
        self.is_fitted = False
        self.file_type = ".csv"
        super().__init__()

    def fit(self, X, y, **kwargs):
        """Fit a linear model using least squares.

        Args:
            X (pd.DataFrame): Data on features.
            y (pd.Series or np.ndarray): Data on outcomes.
            **kwargs:
                - 'fit_intercept' (bool): Should an intercept be fitted.
                - 'n_jobs' (int): Number of jobs to use for parallelization.

        Returns:
            self: The fitted LinearRegression object,

        """
        coefficients = _fit(X=X, y=y, **kwargs)
        self.coefficients = coefficients
        self.is_fitted = True

        return self

    def predict(self, X):
        """Predict outcome using the fitted model and new data.

        Uses the fitted coefficients in ``self.coefficients`` and new data
        ``X`` to predict outcomes. If the model has not been fitted yet we
        return None and a warning.

        Args:
            X (pd.DataFrame): New data on features.

        Returns:
            - None, if ``self.is_fitted`` is False and otherwise
            - predictions (np.array): The predicted outcomes.

        """
        if not self.is_fitted:
            warnings.warn("The model has not been fitted yet.", UserWarning)
            return None

        predictions = _predict(coefficients=self.coefficients, X=X)
        return predictions

    def save(self, filename, overwrite=False):
        """Save fitted model to disc.

        Save the fitted coefficents stored under ``self.coefficients`` to disc
        as a csv file. Can be loaded afterwards using the method ``load``.

        Args:
            filename (str): File path.
            overwrite (bool): Should the file be overwritten if it exists.

        Returns:
            None

        """
        _save(coefficients=self.coefficients, filename=filename, overwrite=overwrite)

    def load(self, filename):
        """Load a fitted model from disc.

        Load fitted coefficients stored as a csv from disc and store them under
        ``self.coefficients``. Will work on files created by method ``save``,
        but also files manually created with first named column "coefficient"
        and second column "value".

        Args:
            filename (str): File path.

        Returns:
            None

        """
        coefficients = _load(filename)
        self.coefficients = coefficients
        self.is_fitted = True

        return self

    @property
    def name(self):
        """Return name of class as string."""
        return self.__class__.__name__


def _fit(X, y, fit_intercept=True, n_jobs=1):
    """Fit a linear model using least squares.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        fit_intercept (bool): Should an intercept be fitted.
        'n_jobs' (int): Number of jobs to use for parallelization.

    Returns:
        coefficients (pd.DataFrame): The named coefficient values.

    """
    assert_input_fit(X=X, y=y)
    XX = np.array(X)

    lm = LinReg(fit_intercept=fit_intercept, n_jobs=n_jobs)
    lm = lm.fit(X=XX, y=y)

    coef_values = lm.coef_.reshape((-1,))
    coef_names = X.columns.tolist()
    if fit_intercept:
        coef_values = np.insert(coef_values, 0, lm.intercept_)
        coef_names = ["intercept"] + coef_names

    coefficients = pd.DataFrame(
        data=zip(coef_names, coef_values), columns=["coefficient", "value"]
    ).set_index("coefficient")

    return coefficients


def _predict(coefficients, X):
    """Predict outcome using the fitted model and new data.

    Args:
        coefficients (pd.DataFrame): The named coefficient values.
        X (pd.DataFrame): New data on features.

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    _assert_input_predict(coefficients, X)

    coef = coefficients[1:]  # drop intercept
    coef = coef.loc[X.columns, :]  # sort coefficients according to data

    predictions = coefficients.loc["intercept", "value"] + X.dot(coef)

    predictions = predictions.values.reshape((-1,))
    return predictions


def _assert_input_predict(coefficients, X):
    """

    Args:
        coefficients:
        X:

    Returns:

    """
    return True


def _save(coefficients, filename, overwrite):
    """Save fitted model to disc.

    Save the fitted coefficents to disc as a csv file.

    Args:
        coefficients (pd.DataFrame): The named coefficient values.
        filename (str): File path.
        overwrite (bool): Should the file be overwritten if it exists.

    Returns:
        None

    """
    file_present = glob.glob(filename)
    if overwrite or not file_present:
        coefficients.to_csv(filename)
    else:
        warnings.warn("File already exists. No actions taken.", UserWarning)


def _load(filename):
    """Load a fitted model from disc.

     Load fitted coefficients stored as a csv from disc and return them. Will
     work on files created by method ``_save``, but also files manually created
     with first named column "coefficient", which will be set as the index.
     For correct functionality of the other functions the second column needs
     to be called "value".

    Args:
        filename (str): File path.

    Returns:
        coefficients (pd.DataFrame): The named coefficient values.

    Raises:
        ValueError, if index column name is wrongly specified in csv file.
        AssertionError, if value column name is incorrect.

    """
    try:
        coefficients = pd.read_csv(filename, index_col="coefficient")
        assert set(coefficients.columns) == {"value"}
    except ValueError:
        raise ValueError(
            "Columns have wrong names. One column has to be"
            "called 'coefficient' containing coefficient names,"
            "and one column has to be called 'value', containing"
            "coefficient values."
        )
    return coefficients
