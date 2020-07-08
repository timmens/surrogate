"""Predict outcomes of test set using the fitted models and save predictions.

TODO: predict kwargs
"""
import pickle

import click
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend
from sklearn.metrics import mean_absolute_error

import src.surrogates as surrogate
from bld.project_paths import project_paths_join as ppj
from src.specs import model_kwargs_to_string
from src.specs import Specification  # noqa: F401
from src.utilities.utilities import load_data
from src.utilities.utilities import subset_features


def _predict(model, specifications, ordered_features, X):
    """Predict model on test data for all model specifications.

    Args:
        model (str): (Economic) simulation model. Must be in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended'].
        specifications (list): List of namedtuple model specifcations with entries
            'model', 'identifier', 'data_kwargs', 'model_kwargs', corresponding to the
            specifications for ``model``.
        ordered_features (pd.Index): Feature index ordered according to some measure of
            importance. See [...].
        X (pd.DataFrame): Testing data.

    Returns:
        predictions (pd.DataFrame): Data frame with columns corresponding to the model
            specifications in ``specifications`` and rows corresponding to the unit
            predictions for each row of ``X``.

    """

    def to_parallelize(spec, model, ordered_features, X):
        order_features = bool(spec.data_kwargs["order_features"])
        n_features = spec.data_kwargs["n_features"]

        XX = subset_features(X, order_features, ordered_features, n_features)

        predictor_path = ppj("OUT_FITTED_MODELS", f"{model}/{spec.identifier}")
        predictor = surrogate.load(predictor_path)

        prediction = surrogate.predict(XX, predictor)
        prediction = pd.Series(prediction.flatten(), name=spec.identifier)
        return prediction

    with parallel_backend("threading", n_jobs=4):
        predictions = Parallel()(
            delayed(to_parallelize)(spec, model, ordered_features, X)
            for spec in specifications
        )

    predictions = pd.concat(predictions, axis=1)
    predictions = predictions.sort_index(axis=1)
    return predictions


def _evaluate(predictions, true, specifications, metrics=None):
    """Evaluate prediction error using various metrics.

    Args:
        predictions (pd.DataFrame): Data frame with columns corresponding to the model
            specifications in ``specifications`` and rows corresponding to the unit
            predictions for each row of ``X``.
        true (pd.DataFrame): The true outcomes.
        specifications (list): List of namedtuple model specifcations with entries
            'model', 'identifier', 'data_kwargs', 'model_kwargs', corresponding to the
            specifications for ``model``.
        metrics (dict): Dictionary containing metrics, that is, functions that may be
            evaluated on an array ``true`` and ``predicted`` which result in a scalar
            output. Example: ``sklearn.metrics.mean_absolute_error``. Defaults to
            {"mae": sklearn.metrics.mean_absolute_error}.

    Returns:
        losses (pd.DataFrame):

    """
    losses = pd.DataFrame(
        columns=["n_features", "n_obs", "order_features", "model", "kwargs", "mae"]
    )
    losses = losses.set_index(
        ["n_features", "n_obs", "order_features", "model", "kwargs"]
    )

    for spec in specifications:
        prediction = predictions[spec.identifier]
        loss = _compute_loss(true, prediction, metrics)

        kwargs = model_kwargs_to_string(spec.model, spec.model_kwargs)
        row = (
            spec.data_kwargs["n_features"],
            spec.data_kwargs["n_obs"],
            spec.data_kwargs["order_features"],
            spec.model,
            kwargs,
        )

        losses.loc[row, "mae"] = loss

    losses = losses.sort_index()
    return losses


def _compute_loss(true, predicted, metrics=None):
    """Compute loss of prediction given various metrics.

    Args:
        true (np.array): True outcome.
        predicted (np.array): Predicted outcome.
        metrics (dict): Dictionary containing metrics, that is, functions that may be
            evaluated on an array ``true`` and ``predicted`` which result in a scalar
            output. Example: ``sklearn.metrics.mean_absolute_error``. Defaults to
            {"mae": sklearn.metrics.mean_absolute_error}.

    Returns:
        loss (pd.Series or float): The occured loss from prediction. If multiple metrics
            are passed a pd.Series object is returned and float otherwise.

    """
    metrics = {"mae": mean_absolute_error} if metrics is None else metrics
    loss = [metric(true, predicted) for metric in metrics.values()]
    loss = pd.Series(loss, index=metrics.keys())
    loss = loss if len(loss) > 1 else float(loss)
    return loss


@click.command()
@click.argument("model", type=str)
def main(model):
    # predict using fitted models
    with open(ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl"), "rb") as f:
        specifications = pickle.load(f)

    try:
        ordered_features = pd.read_csv(ppj("OUT_DATA", f"{model}-ordered_features.csv"))
    except FileNotFoundError:
        ordered_features = None

    X, y = load_data(model, testing=True)

    predictions = _predict(model, specifications, ordered_features, X)

    # evaluate model predictions
    losses = _evaluate(predictions, y, specifications)

    # save
    prediction_path = ppj("OUT_ANALYSIS", f"{model}-predictions.pkl")
    predictions.to_pickle(prediction_path)

    loss_path = ppj("OUT_ANALYSIS", f"{model}-losses.csv")
    losses.to_csv(loss_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
