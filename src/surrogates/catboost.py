"""Function wrapper to use catboost for regression."""
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utilities import suppress_stdout


def fit(X, y, iterations=1_000, learning_rate=0.05, depth=8, loss_function="MAE"):
    """Fit a boosted tree using catboost.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        iterations (int): Number of models to build.
        learning_rate (float): Learning rate for updating step.
        depth (int): Depth of single model. Represent depth of oblivious trees.
        loss_function (str): Loss function over which the model is optimized.

    Returns:
        predictor (dict): Dictionary with entries 'model' for the fitted model and
            'pipe' for the pre-processing pipeline.
            - model : catboost.CatBoostRegressor
            - pipe : sklearn.pipeline.Pipeline

    """
    preprocess_steps = [("scale", StandardScaler())]
    pipe = Pipeline(preprocess_steps)

    XX = pipe.fit_transform(X)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
    )
    with suppress_stdout():
        model.fit(XX, y)

    predictor = {"model": model, "pipe": pipe}
    return predictor


def predict(X, predictor):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        predictor (dict): Dictionary with entries 'model' for the fitted model and
            'pipe' for the pre-processing pipeline.
            - model : catboost.CatBoostRegressor
            - pipe : sklearn.pipeline.Pipeline

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    XX = predictor["pipe"].transform(X)
    predictions = predictor["model"].predict(XX)
    return predictions
