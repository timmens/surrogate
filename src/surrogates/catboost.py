"""Function wrapper to use catboost for regression."""
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

from src.shared import get_model_and_kwargs_from_type
from src.surrogates.polynomialregression import fit as pre_fit
from src.surrogates.polynomialregression import predict as pre_predict


def fit(
    X,
    y,
    iterations=1_000,
    learning_rate=0.05,
    depth=8,
    loss_function="MAE",
    pre_fit_model=None,
):
    """Fit a boosted tree using catboost.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        iterations (int): Number of models to build.
        learning_rate (float): Learning rate for updating step.
        depth (int): Depth of single model. Represent depth of oblivious trees.
        loss_function (str): Loss function over which the model is optimized.
        pre_fit_model (str): Name of model that should be fit in a first stage. The
            catboost regressor is then trained on the residuals. If None, then the
            catboost model is fit directly. Has to be a model which used the module
            ``polynomialregression``.

    Returns:
        predictor (dict): Dictionary with entries 'model' for the fitted model and
            'pipe' for the pre-processing pipeline.
            - model : catboost.CatBoostRegressor
            - pipe : sklearn.preprocessing.StandardScaler
            - pre_fit_model : None or pre fitted model.

    """
    pipe = StandardScaler()
    XX = pipe.fit_transform(X)

    if pre_fit_model is None:
        residuals = y
    else:
        _, kwargs = get_model_and_kwargs_from_type(pre_fit_model)
        pre_fit_model = pre_fit(XX, y, **kwargs)
        residuals = y - pre_predict(XX, pre_fit_model)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
    )
    model.fit(XX, residuals, verbose=False)

    predictor = {"model": model, "pipe": pipe, "pre_fit_model": pre_fit_model}
    return predictor


def predict(X, predictor):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        predictor (dict): Dictionary with entries 'model' for the fitted model and
            'pipe' for the pre-processing pipeline.
            - model : catboost.CatBoostRegressor
            - pipe : sklearn.preprocessing.StandardScaler
            - pre_fit_model : None or pre fitted model.

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    XX = predictor["pipe"].transform(X)

    predictions = predictor["model"].predict(XX)

    pre_fit_model = predictor["pre_fit_model"]
    if pre_fit_model is not None:
        predictions += pre_predict(XX, pre_fit_model).flatten()

    return predictions
