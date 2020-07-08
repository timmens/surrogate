"""Generate model specifications.

Generate a list of named tuples which represent model specifications that should be
run using the specifications file ``src/specs/specifications.json``.
"""
import json
import pickle
from collections import namedtuple

import click
import pandas as pd
import scipy.special

from bld.project_paths import project_paths_join as ppj


Specification = namedtuple(
    "Specification", ["model", "identifier", "data_kwargs", "model_kwargs"]
)


def _create_specifications(models, model_kwargs, n_features, n_obs, order_features):
    data_kwargs_names = ["model", "n_features", "n_obs", "order_features"]

    model_specs = []
    for model in models:
        kwargs = model_kwargs[model]
        model_kwargs_names = kwargs.keys()

        combinations = _create_combinations(
            model, n_features, n_obs, order_features, kwargs
        )
        combinations = combinations.apply(
            lambda row: _series_to_namedtuple(
                row, model, data_kwargs_names, model_kwargs_names
            ),
            axis=0,
        )
        model_specs.extend(combinations)

    specifications = [specs for specs in model_specs if _is_valid(specs)]
    return specifications


def _create_combinations(model, n_features, n_obs, order_features, kwargs):
    """Create data frame with cross-product combinations.

    Since different models use different (extra) keyword arguments, here, we flexibly
    add potentially multiple keyword arguments to the range of cross-product
    combinations.

    Args:
        model (str): Model name.
        n_features (list): List of number of features to use.
        n_obs (list): List of number of observations to use.
        order_features (list): List of 0 or 1, denoting wether to use ordered features
            or random feature selection when using less features than are available.
        kwargs (dict): kwargs with list of extra arguments passed to model specific
            fitting function. Number of values can be more than 1.

    Returns:
        combinations (pd.DataFrame): Data frame in which each row represents one
            parameter combination.

    """
    kwargs_names = list(kwargs.keys())
    kwargs_values = list(kwargs.values())

    to_product = [[model], n_features, n_obs, order_features] + kwargs_values
    combinations = pd.core.reshape.util.cartesian_product(to_product)

    index = ["model", "n_features", "n_obs", "order_features"] + kwargs_names
    combinations = pd.DataFrame(combinations, index=index)
    return combinations


def _series_to_namedtuple(row, model, data_kwargs_names, model_kwargs_names):
    """Convert pd.Series to namedtuple.

    Args:
        model (str): Model name.
        row (pd.Series): Series containing corresponding values.
        data_kwargs_names (list): List of index names (of row) for data_kwargs.
        model_kwargs_names (list): List of index names (of row) for model kwargs.

    Returns:
        spec (namedtuple.Specification): Specification named tuple.
    """
    data_kwargs = row[data_kwargs_names].to_dict()
    model_kwargs = row[model_kwargs_names].to_dict()

    data_identifier = _model_args_identifier(**data_kwargs)
    model_identifier = model_kwargs_to_string(model, model_kwargs)
    identifier = data_identifier + "_" + model_identifier

    spec = Specification(
        model=model,
        identifier=identifier,
        data_kwargs=data_kwargs,
        model_kwargs=model_kwargs,
    )
    return spec


def _model_args_identifier(model, n_features, n_obs, order_features):
    """Create unique model identifier given standard arguments.

    Args:
        model (str): Model name.
        n_features (int): Number of features used in training.
        n_obs (int): Number of observations used in training.
        order_features (bool): Should the subset of features be selected with resepect
            to some ordering.

    Returns:
        identifier (str): Unique identifier name.

    """
    order_features = 1 if order_features else 0
    identifier = f"{model}_p{n_features}_n{n_obs}_o{order_features}"
    return identifier


def model_kwargs_to_string(model, kwargs):
    """Create unique model identifier given additional keyword arguments.

    Args:
        model (str): Model name.
        kwargs (dict): kwargs with list of extra arguments passed to model specific
            fitting function. Number of values can be more than 1.

    """
    if model in ["polynomial", "ridge"]:
        degree = kwargs["degree"]
        identifier = f"degree{degree}"
    elif model in ["neuralnetwork"]:
        layers = "-".join(kwargs["layers"])
        identifier = f"layers{layers}"
    else:
        raise NotImplementedError
    return identifier


def _is_valid(specification):
    """Check if model specification is valid.

    Some specifications can turn out to be invalid. This functions helps in finding
    them. For example, a 2nd degree polynomial model with 10 features cannot be fitted
    using less than 45 observations.

    Args:
        specification (namedtuple): Named tuple with entries "model", "identifier",
            "data_kwargs" and "model_kwargs".


    Returns:
        valid (bool): True if the specification is valid else False.
    """
    if "polynomial" in specification.identifier:
        degree = specification.model_kwargs["degree"]
        n_features = specification.data_kwargs["n_features"]
        n_obs = specification.data_kwargs["n_obs"]
        valid = n_obs >= scipy.special.binom(n_features, degree) + 1
    elif "neuralnet" in specification.identifier:
        layers = specification.model_kwargs["layers"]
        n_layers = len(layers)
        valid = n_obs >= 1000 * n_layers
    else:
        valid = True
    return valid


@click.command()
@click.argument("model", type=str)
def main(model):
    data_path = ppj("IN_MODEL_SPECS", "specifications.json")
    with open(data_path) as handle:
        raw_specs = json.loads(handle.read())

    specs = raw_specs[model]
    specifications = _create_specifications(**specs)
    file_name = ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl")
    with open(file_name, "wb") as handle:
        pickle.dump(specifications, handle)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
