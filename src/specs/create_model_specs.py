"""Generate model specifications.

Generate a list of named tuples which represent model specifications that should be
run using the specifications file ``src/specs/specifications.json``.
"""
import json

import click
import cloudpickle
import pandas as pd
from scipy.special import binom

from bld.project_paths import project_paths_join as ppj
from src.specs.specification import Specification


def _create_specifications(models, n_obs, fit_kwargs, predict_kwargs, n_features):
    """Create list of specifications to run given models, observations and model kwargs.

    Args:
        model (str): (Surrogate) Model name.
        n_obs (list): List of number of observations to use.
        fit_kwargs (dict): kwargs with list of extra arguments passed to model specific
            fitting function for each surrogate model.
        predict_kws (dict): kwargs with list of extra arguments passed to model
            specific predicting function for each surrogate model.
        n_features (int): Number of input features in simulation model. Used to check
            wether specifications are valid.

    Returns:
        specifications (list): List of namedtuple of form ``Specification``, containing
            the specification parameter combinations.

    """
    specifications = []
    for model in models:
        try:
            fit_kws = fit_kwargs[model]
        except KeyError:
            fit_kws = {}
        try:
            predict_kws = predict_kwargs[model]
        except KeyError:
            predict_kws = {}

        combinations = _create_combinations(model, n_obs, fit_kws, predict_kws)
        specifications.extend(combinations)

    specifications = [spec for spec in specifications if _is_valid(spec, n_features)]
    return specifications


def _create_combinations(model, n_obs, fit_kws, predict_kws):
    """Create data frame with cross-product combinations.

    Since different models use different (extra) keyword arguments, here, we flexibly
    add potentially multiple keyword arguments to the range of cross-product
    combinations.

    Args:
        model (str): (Surrogate) Model name.
        n_obs (list): List of number of observations to use.
        fit_kws (dict): kwargs with list of extra arguments passed to model specific
            fitting function. Number of values can be more than 1.
        predict_kws (dict): kwargs with list of extra arguments passed to model
            specific predicting function. Number of values can be more than 1.

    Returns:
        combinations (list): List of namedtuple of form ``Specification``, containing
            the specification parameter combinations.

    """
    to_product = [[model], n_obs] + list(fit_kws.values()) + list(predict_kws.values())
    combinations = pd.core.reshape.util.cartesian_product(to_product)

    fit_kws_names = list(fit_kws.keys())
    predict_kws_names = list(predict_kws.keys())

    index = ["model", "n_obs"] + fit_kws_names + predict_kws_names
    combinations = pd.DataFrame(combinations, index=index)

    combinations = combinations.apply(
        lambda row: _series_to_namedtuple(row, fit_kws_names, predict_kws_names),
        axis=0,
    )
    return combinations


def _series_to_namedtuple(row, fit_kws_names, predict_kws_names):
    """Convert specification pd.Series to namedtuple.

    Args:
        row (pd.Series): Series containing corresponding specification values.
        fit_kws_names (list): List of index names (of row) for fit_kws.
        predict_kws_names (list): List of index names (of row) for predict_kws.

    Returns:
        spec (namedtuple.Specification): Specification named tuple.

    """
    fit_kws = row[fit_kws_names].to_dict()
    predict_kws = row[predict_kws_names].to_dict()
    model = row["model"]
    n_obs = row["n_obs"]

    fit_kws_id = "_".join((f"{v}_{k}" for v, k in fit_kws.items()))
    if len(predict_kws) == 0:
        predict_kws_id = "None"
    else:
        predict_kws_id = "_".join((f"{v}_{k}" for v, k in predict_kws.items()))

    identifier = f"{model}_n-{n_obs}_f_{fit_kws_id}_p_{predict_kws_id}"
    identifier = identifier.replace(".", "")

    spec = Specification(
        model=model,
        n_obs=n_obs,
        identifier=identifier,
        fit_kwargs=fit_kws,
        predict_kwargs=predict_kws,
    )
    return spec


def _is_valid(specification, n_features):
    """Check if model specification is valid.

    Some specifications can turn out to be invalid or inefficient. This functions helps
    in finding them. Examples:

    - A neural network with 10 layers fitted on 1000 observations is not expected to
        recover much information, so we drop it.
    - A polynomial model of degree 2 with 10 input features has no unique solution with
        less observations than binom(10, 2) + 21

    Args:
        specification (namedtuple): Named tuple with entries "model", "identifier",
            "fit_kwargs" and "predict_kwargs".
        n_features (int): Number of input features of simulation model.

    Returns:
        valid (bool): True if the specification is valid else False.

    """
    n_obs = specification.n_obs
    if "neuralnet" == specification.model:
        layers = specification.fit_kwargs["layers"]
        valid = n_obs >= 1000 * len(layers)
    elif "polynomial" == specification.model:
        degree = specification.fit_kwargs["degree"]
        valid = n_obs >= _number_coefficients_polynomial_model(n_features, degree)
    else:
        valid = True
    return valid


def _number_coefficients_polynomial_model(n_features, degree):
    """Return number of coefficients in polynomial model with interactions.

    Args:
        n_features (int): Number of input features in simulation model.
        degree (int): Number of degrees of polynomial.

    Returns:
        n_coefficients (int): Number of coefficients.

    """

    def combinations(k):
        return int(binom(n_features - 1 + k, n_features - 1))

    n_coefficients = sum(combinations(k) for k in range(degree + 1))
    return n_coefficients


@click.command()
@click.argument("model", type=str)
def main(model):
    spec_path = ppj("IN_MODEL_SPECS", f"{model}-specification.json")
    with open(spec_path) as spec_file:
        specs = json.loads(spec_file.read())

    specifications = _create_specifications(**specs)

    file_name = ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl")
    with open(file_name, "wb") as handle:
        cloudpickle.dump(specifications, handle)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
