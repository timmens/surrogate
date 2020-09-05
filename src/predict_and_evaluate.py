"""Predict outcomes of test set using the fitted surrogates and save predictions."""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

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
        for name, fitted_surrogate in tqdm(fitted_surrogates.items())
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
        losses.index.str.split("_n-").to_list(), names=["model", "n_obs"]
    )
    return losses


def _save_results(predictions, losses, path):
    """Save results to pickle and csv."""
    predictions.to_pickle(path / "predictions.pkl")
    losses.to_csv(path / "losses.csv")


def _load_fitted_surrogates(project_name):
    """Load fitted surrogate models for given project."""
    path = BLD / "surrogates" / project_name
    surrogate_paths = path.glob("*.pkl")
    fitted_surrogates = {p.stem: surrogates.load(p) for p in surrogate_paths}
    return fitted_surrogates


def main():
    specifications = read_specifications()
    for project_name, spec in specifications.items():
        fitted_surrogates = _load_fitted_surrogates(project_name)
        predictions = _predict(spec["data_set"], fitted_surrogates)
        losses = _evaluate(spec["data_set"], predictions)

        build_path = BLD / "evaluations" / project_name
        build_path.mkdir(parents=True, exist_ok=True)
        _save_results(predictions, losses, build_path)


if __name__ == "__main__":
    main()
