"""Predict outcomes of test set using the fitted surrogates and save predictions."""
import pandas as pd
import pytask
from sklearn.metrics import mean_absolute_error

import src.surrogates as surrogates
from src.config import BLD
from src.shared import load_data
from src.specs import read_specifications


def _predict(data_set, fitted_surrogates):
    """Predict model on test data for all model specifications.

    Returns:
        predictions (pd.DataFrame): Data frame with columns corresponding to the model
            specifications in ``specifications`` and rows corresponding to the unit
            predictions for each row of ``X``.

    """
    X_test, _ = load_data(data_set, testing=True)
    predictions = {
        name: surrogates.predict(X_test, fitted_surrogate).flatten()
        for name, fitted_surrogate in fitted_surrogates.items()
    }
    predictions = pd.DataFrame.from_dict(predictions).sort_index(axis=1)
    return predictions


def _evaluate(data_set, predictions):
    """Evaluate prediction error using mean absolute error.

    Args:
        data_set (str):
        predictions (pd.DataFrame):

    Returns:
        losses (pd.DataFrame): MAE losses for each surrogate model on test set.

    """
    _, y_test = load_data(data_set, testing=True)

    losses = predictions.apply(lambda col: mean_absolute_error(y_test, col), axis=0)
    losses = pd.DataFrame(losses, columns=["mae"])
    losses.index = pd.MultiIndex.from_tuples(
        losses.index.str.split("-").to_list(), names=["data_set", "model", "n_obs"]
    )
    losses.index = losses.index.droplevel(0)
    return losses


def _load_fitted_surrogates(project_name):
    """Load fitted surrogate models for given project."""
    path = BLD / "surrogates" / project_name
    surrogate_paths = path.glob("*.pkl")
    fitted_surrogates = {p.stem: surrogates.load(p) for p in surrogate_paths}
    return fitted_surrogates


def load_specifications():
    specifications = read_specifications(fitting=True)
    project_names = list(specifications.keys())
    produces = [
        (
            BLD / "evaluations" / project_name / "predictions.pkl",
            BLD / "evaluations" / project_name / "losses.csv",
        )
        for project_name in project_names
    ]
    return zip(produces, project_names)


def _load_data_set(project_name):
    specifications = read_specifications()
    data_set = specifications[project_name]["data_set"][0]
    return data_set


@pytask.mark.parametrize("produces, project_name", load_specifications())
def task_predict_and_evaluate(produces, project_name):
    fitted_surrogates = _load_fitted_surrogates(project_name)
    data_set = _load_data_set(project_name)
    predictions = _predict(data_set, fitted_surrogates)
    losses = _evaluate(data_set, predictions)

    predictions.to_pickle(produces[0])
    losses.to_csv(produces[1])
