"""Predict outcomes of test set using the fitted models and save predictions."""
import pickle

import click
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

import src.surrogates as surrogate
from bld.project_paths import project_paths_join as ppj
from src.specs import Specification  # noqa: F401
from src.utilities import load_data


def _predict(simulation_model, specifications, X):
    """Predict model on test data for all model specifications.

    Args:
        simulation_model (str): (Economic) simulation model. Must be in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended'].
        specifications (list): List of namedtuple model specifcations with entries
            'model', 'identifier', 'fit_kwargs', 'predict_kwargs', corresponding to the
            specifications for ``model``.
        X (pd.DataFrame): Testing data.

    Returns:
        predictions (pd.DataFrame): Data frame with columns corresponding to the model
            specifications in ``specifications`` and rows corresponding to the unit
            predictions for each row of ``X``.

    """

    def to_parallelize(spec, simulation_model=simulation_model, X=X):
        predictor_path = ppj(
            "OUT_FITTED_MODELS", f"{simulation_model}/{spec.identifier}"
        )
        predictor = surrogate.load(predictor_path)

        prediction = surrogate.predict(X, predictor, **spec.predict_kwargs)
        prediction = pd.Series(prediction.flatten(), name=spec.identifier)
        return prediction

    with parallel_backend("threading", n_jobs=4):
        predictions = Parallel()(
            delayed(to_parallelize)(spec) for spec in tqdm(specifications)
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
    losses = pd.DataFrame(columns=["model", "n_obs", "kwargs", "mae"])
    losses = losses.set_index(["model", "n_obs", "kwargs"])

    for spec in specifications:
        prediction = predictions[spec.identifier]
        loss = _compute_loss(true, prediction, metrics)

        model_kwargs_id = _model_kwargs_to_string(spec)
        losses.loc[(spec.model, spec.n_obs, model_kwargs_id), "mae"] = loss

    losses = losses.sort_index()
    return losses


def _model_kwargs_to_string(spec):
    """Return for a given specification a string identifying using its kwargs.

    Args:
        spec (namedtuple): Namedtuple of type ``Specification`` with entries 'model',
            'identifier', 'n_obs', 'fit_kwargs', 'predict_kwargs'.

    Returns:
        model_kwargs_id (str): String identifying a model in a specific sub-model class.

    """
    model_kwargs_id = "-".join(spec.identifier.split("_")[2:])
    return model_kwargs_id


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

    X, y = load_data(model, testing=True)
    predictions = _predict(model, specifications, X)

    # evaluate model predictions
    losses = _evaluate(predictions, y, specifications)

    # save
    predictions.to_pickle(ppj("OUT_ANALYSIS", f"{model}-predictions.pkl"))
    losses.to_csv(ppj("OUT_ANALYSIS", f"{model}-losses.csv"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
