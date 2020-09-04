"""Load specific model and fit to training data set."""
import itertools

import src.surrogates as surrogates
from src.config import BLD
from src.shared import get_model_and_kwargs_from_type
from src.shared import load_data
from src.specs import read_specifications


def fit_surrogates(data_set, surrogates, n_obs):
    """Fit all models specified in ``specifications`` and save to disc.

    Args:
        data_set (list): List of names of data sets which shall be used for the
            training procedure. Must be a subset of ['kw_94_one', 'kw_97_basic',
            'kw_97_extended'].
        surrogates (list): List of names of surrogate models which shall be
            used for fitting. Elements must be specified in
            ``src/surrogates/name_to_kwargs.yaml``.
        n_obs (list): List of number of observations to be use.

    Returns:
        predictors_and_id (dict): Dictionary containing fitted regression models. Keys
            are the model id and values the respective models.

    """
    predictors_and_id = {}
    for surrogate_type, n in itertools.product(surrogates, n_obs):
        X, y = load_data(data_set, n_train=n)
        predictor = _fit_internal(X, y, surrogate_type, n_obs)
        id_ = _make_id(surrogate_type, n)
        predictors_and_id[id_] = predictor

    return predictors_and_id


def save_surrogates(predictors_and_id, path):
    """Save models to path.

    Args:
        predictors_and_id (dict): Dictionary containing fitted regression models. Keys
            are the model id and values the respective models.
        path (pathlib.Path): Path where to save the models to.

    Returns:
        None

    """
    for name, model in predictors_and_id.items():
        surrogates.save(model, path / name, overwrite=True)


def _fit_internal(X, y, surrogate_type, n_obs):
    """Fit all models specified in ``specifications`` and save to disc.

    Args:
        data_set (str): Name of data set which is used for the training procedure. Must
            be in ['kw_94_one', 'kw_97_basic', 'kw_97_extended'].
        surrogate_type (str): Name of surrogate model which will be used for fitting.
            Must be specified in ``src/surrogates/name_to_kwargs.yaml``.

    Returns:
        predictor (dict): Dictionary containing 'model' and 'pipe'. Model is of type
            ``surrogate_type``.

    """
    model, kwargs = get_model_and_kwargs_from_type(surrogate_type)
    predictor = surrogates.fit(model, X, y, **kwargs)
    return predictor


def _make_id(surrogate_type, n_obs):
    """Make ID from specifications."""
    id_ = f"{surrogate_type}_n-{n_obs}"
    return id_


def main():
    specifications = read_specifications()
    for project, spec in specifications.items():
        predictors_and_id = fit_surrogates(**spec)
        build_path = BLD / "surrogates" / project
        build_path.mkdir(parents=True, exist_ok=True)
        save_surrogates(predictors_and_id, build_path)


if __name__ == "__main__":
    main()
