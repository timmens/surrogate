"""Function wrapper to perform ridge regression."""
from collections import namedtuple
from copy import deepcopy

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


RidgePredictor = namedtuple("RidgePredictor", ["model", "pipe"])


def fit(X, y, degree=1, fit_intercept=True, scale=True, alphas=None, cv=5, n_jobs=1):
    """Fit and return a (polynomial) ridge regression model.

    Args:
        X (pd.DataFrame, pd.Series, np.ndarary): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        degree (int): Degree of the polynomial model. Default is 1.
        fit_intercept (bool): Should an intercept be fitted. Default is True.
        scale (bool): Scale data before performing regression. Default is True.
        alphas (np.ndarray): Array of dimension 1 containing the alpha values on which
            we perform cross-validation. Defaults to ``np.logspace(-5, 1, 100)``.
        cv (int): Number of folds used during cross-validation.
        n_jobs (int): Number of jobs to use for parallelization. Default is 1.

    Returns:
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline.
                model : sklearn.linear_model.LinearRegression
                pipe : sklearn.pipeline.Pipeline

    """
    preprocess_steps = [("poly", PolynomialFeatures(degree=degree, include_bias=False))]
    if scale:
        preprocess_steps += [("scale", StandardScaler())]

    pipe = Pipeline(preprocess_steps)
    XX = pipe.fit_transform(X)

    alphas = np.insert(np.logspace(-5, 1, 100), 0, 0) if alphas is None else alphas
    rr = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, cv=cv)
    rr = rr.fit(X=XX, y=y)

    predictor = RidgePredictor(rr, pipe)
    return predictor


def predict(X, predictor, threshold=None):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline.
                model : sklearn.linear_model.LinearRegression
                pipe : sklearn.pipeline.Pipeline
        threshold (float): Coefficients below threshold are set to zero. Default is
            np.inf.

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    threshold = np.inf if threshold is None else threshold
    predictor = deepcopy(predictor)

    XX = predictor.pipe.transform(X)

    coef = predictor.model.coef_.copy()
    mask = np.abs(coef) < threshold
    predictor.model.coef_[mask] = 0

    predictions = predictor.model.predict(XX)
    return predictions
