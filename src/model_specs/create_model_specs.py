import json

from bld.project_paths import project_paths_join as ppj


def _create_model_specs_varying_samples(
    model_prefix, model, nfeatures, nobs_list, kwargs,
):
    """Create models specs for many models with obs in nobs_list.

    Args:
        model_prefix (str): Prefix used to identify model.
        model (str): Model name.
        nfeatures (int): Number of features used in training.
        nobs_list (list): List of different number of observations
            used in training.
        kwargs (dict): Kwargs used in fitting of model.

    Returns:

    """
    model_specs = {}
    for nobs in nobs_list:
        name = _create_unique_model_name(model_prefix, nfeatures, nobs)
        model_specs[name] = _create_specs(model, nfeatures, nobs, kwargs)
    return model_specs


def _create_model_specs_varying_features(
    model_prefix, model, nfeatures_list, nobs, kwargs,
):
    """Create model specs for many models with features in nfeatures_list.

    Args:
        model_prefix (str): Prefix used to identify model.
        model (str): Model name.
        nfeatures_list (list): List of different number of
            features for which specs should be created.
        nobs (int): Number of observations used in training.
        kwargs (dict): Kwargs used in fitting of model.

    Returns:

    """
    model_specs = {}
    for nfeatures in nfeatures_list:
        name = _create_unique_model_name(model_prefix, nfeatures, nobs)
        model_specs[name] = _create_specs(model, nfeatures, nobs, kwargs)
    return model_specs


def _create_unique_model_name(model, nfeatures, nobs):
    """Create unique model identifier name.

    Args:
        model (str): Model name.
        nfeatures (int): Number of features used in training.
        nobs (int): Number of observations used in training.

    Returns:
        name (str): Unique identifier name.

    """
    name = f"{model}_p{nfeatures}_n{nobs}"
    return name


def _create_specs(model, nfeatures, nobs, kwargs):
    """Create specifications for single model.

    Args:
        model (str): Model name.
        nfeatures (int): Number of features used in training.
        nobs (int): Number of observations used in training.
        kwargs (dict): Kwargs used in fitting of model.

    Returns:
        specs (dict): Model specifications in dictionary.

    """
    specs = {"model": model, "nfeatures": nfeatures, "nobs": nobs, "kwargs": kwargs}
    return specs


if __name__ == "__main__":

    # varying number of features
    prefix = ["linreg", "polreg", "ridgereg"]
    models = ["polynomialregression", "polynomialregression", "ridgeregression"]
    kwargs = [{"degree": 1}, {"degree": 2}, {"degree": 2}]
    nfeat = [5, 10, 15, 20, 25]

    varying_features = {}
    for p, m, k in zip(prefix, models, kwargs):
        varying_features = dict(
            varying_features,
            **_create_model_specs_varying_features(p, m, nfeat, nobs=10000, kwargs=k),
        )

    # varying number of observations
    nobs_list = [100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000]
    varying_observations = {}

    # linreg
    varying_observations = dict(
        varying_observations,
        **_create_model_specs_varying_samples(
            model_prefix="linreg",
            model="polynomialregression",
            nfeatures=27,
            nobs_list=nobs_list,
            kwargs={"degree": 1},
        ),
    )

    # polreg
    varying_observations = dict(
        varying_observations,
        **_create_model_specs_varying_samples(
            model_prefix="polreg",
            model="polynomialregression",
            nfeatures=27,
            nobs_list=nobs_list[2:],  # remove 100, 200 since df(polreg) = 406.
            kwargs={"degree": 2},
        ),
    )

    # ridgereg
    varying_observations = dict(
        varying_observations,
        **_create_model_specs_varying_samples(
            model_prefix="ridgereg",
            model="ridgeregression",
            nfeatures=27,
            nobs_list=nobs_list,
            kwargs={"degree": 2},
        ),
    )

    # nnets
    nobs_list_nnet = nobs_list.copy()
    nobs_list_nnet.extend([25000, 50000])

    layers = [
        [54, 54],
        [81, 81, 81],
        [54, 54, 54, 54, 54],
    ]
    nnet_prefix = ["small", "large", "deep"]

    for layer, p in zip(layers, nnet_prefix):
        varying_observations = dict(
            varying_observations,
            **_create_model_specs_varying_samples(
                model_prefix=f"nnet-{p}",
                model="neuralnetwork",
                nfeatures=27,
                nobs_list=nobs_list_nnet,
                kwargs={"layers": layer, "n_epochs": 100, "n_batch_size": 10},
            ),
        )

    # all specifications
    specs = varying_features.copy()
    specs = dict(specs, **varying_observations)

    with open(ppj("OUT_MODEL_SPECS", "model_specs.json"), "w") as handle:
        json.dump(specs, handle)
    with open(ppj("IN_MODEL_SPECS", "model_specs.json"), "w") as handle:
        json.dump(specs, handle)
