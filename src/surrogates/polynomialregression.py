"""Function wrapper to perform polynomial regression."""
from collections import namedtuple

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


PolynomialPredictor = namedtuple("PolynomialPredictor", ["model", "pipe"])


def fit(X, y, degree=1, fit_intercept=True, scale=True, n_jobs=1):
    """Fit and return a polynomial regression model using least squares.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        degree (int): Degree of the polynomial model. Default is 1.
        fit_intercept (bool): Should an intercept be fitted. Default is True.
        scale (bool): Scale data before performing regression. Default is True.
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

    lm = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
    lm = lm.fit(X=XX, y=y)

    predictor = PolynomialPredictor(lm, pipe)
    return predictor


def predict(X, predictor):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline.
                model : sklearn.linear_model.LinearRegression
                pipe : sklearn.pipeline.Pipeline

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    XX = predictor.pipe.transform(X)
    predictions = predictor.model.predict(XX)
    return predictions
