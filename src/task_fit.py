"""Load model specifications and fit to training data set."""
import itertools
import warnings

import pandas as pd
import pytask

import src.surrogates as surrogate
from src.config import BLD
from src.shared import get_model_and_kwargs_from_type
from src.shared import load_data
from src.specs import read_specifications
from src.task_train_test_split import load_specifications as load_split_specs


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
        fitted_surrogate = surrogate.fit(model, X, y, **kwargs)
    return fitted_surrogate


def load_specifications():
    """Load specifications and return as list of lists."""
    specifications = read_specifications(fitting=True)

    columns = ["data_set", "surrogate_type", "n_obs"]
    df_list = []
    for project_name, spec in specifications.items():
        args = list(itertools.product(*spec.values()))
        df_args = pd.DataFrame(args, columns=columns)
        df_args = df_args.drop_duplicates()
        df_args["produces"] = df_args.apply(
            make_produces_path, axis=1, project_name=project_name
        )
        df_args = df_args[["produces"] + columns]
        df_list += [df_args]

    df = pd.concat(df_list, axis=0)
    args_list = df.values.tolist()
    return args_list


def make_produces_path(args, project_name):
    """Make produces path from args."""
    path = BLD / "surrogates" / project_name
    produces = path / ("-".join(str(a) for a in args) + ".pkl")
    return produces


def load_dependencies():
    dependencies, _ = zip(*load_split_specs())
    return dependencies[0]


@pytask.mark.depends_on(load_dependencies())
@pytask.mark.parametrize(
    "produces, data_set, surrogate_type, n_obs", load_specifications()
)
def task_fit_surrogates(produces, data_set, surrogate_type, n_obs):
    fitted_model = fit_surrogate(data_set, surrogate_type, n_obs)
    surrogate.save(fitted_model, produces)
