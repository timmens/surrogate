"""Load model specifications and fit to training data set."""
import itertools
import warnings

import pandas as pd
import pytask

import src.surrogates as surrogates
from src.config import BLD
from src.shared import get_model_and_kwargs_from_type
from src.shared import load_data
from src.specs import read_specifications


def fit_surrogate(data_set, surrogate_type, n_obs, ignore_warnings=True):
    """Fit all models specified in ``specifications`` and save to disc.

    Args:
        data_set (list): List of names of data sets which shall be used for the
            training procedure. Must be a subset of ['kw_94_one', 'kw_97_basic',
            'kw_97_extended'].
        surrogate_type (str): Names of surrogate model which is used for fitting. Must be
            specified in ``src/surrogates/name_to_kwargs.yaml``.
        n_obs (str): Number of observations to be used.

    Returns:
        fitted_surrgate (dict): Dictionary containing at least 'model' and 'pipe'.
            Model is of type ``surrogate_type``.

    """
    X, y = load_data(data_set, n_train=n_obs)
    model, kwargs = get_model_and_kwargs_from_type(surrogate_type)
    with warnings.catch_warnings():
        warning_filter = "ignore" if ignore_warnings else "default"
        warnings.simplefilter(warning_filter)
        fitted_surrogate = surrogates.fit(model, X, y, **kwargs)
    return fitted_surrogate


def load_specifications():
    """Load specifications and return as list of lists."""
    specifications = read_specifications(fitting=True)

    args = []
    for _, spec in specifications.items():
        args.extend(itertools.product(*spec.values()))

    df_args = pd.DataFrame(args, columns=list(specifications.popitem()[1].keys()))
    df_args = df_args.drop_duplicates()
    df_args["produces"] = df_args.apply(_make_produces_path, axis=1)
    df_args = df_args[["produces", "data_set", "surrogate_type", "n_obs"]]

    args_list = df_args.values.tolist()
    return args_list


def _make_produces_path(args, project_name="paper_uncertainty-propagation"):
    """Make produces path from args."""
    path = BLD / "surrogates" / project_name
    produces = path / ("-".join(str(a) for a in args) + ".pkl")
    return produces


@pytask.mark.parametrize(
    "produces, data_set, surrogate_type, n_obs", load_specifications()
)
def task_fit_surrogates(produces, data_set, surrogate_type, n_obs):
    fitted_model = fit_surrogate(data_set, surrogate_type, n_obs)
    surrogates.save(fitted_model, produces)
