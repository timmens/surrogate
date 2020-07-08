"""Load specific model and fit to training data set."""
import pickle
from pathlib import Path

import click
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend

import src.surrogates as surrogates
from bld.project_paths import project_paths_join as ppj
from src.specs import Specification  # noqa: F401
from src.utilities.utilities import load_data
from src.utilities.utilities import subset_features


def _fit(model, specifications, ordered_features, save_path):
    """Fit all models specified in ``specifications``.

    Args:
        model (str): (Economic) simulation model. Must be in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended'].
        specifications (list): List of namedtuple model specifcations with entries
            'model', 'identifier', 'data_kwargs', 'model_kwargs', corresponding to the
            specifications for ``model``.
        ordered_features (pd.Index): Feature index ordered according to some measure of
            importance. See [...].
        save_path (pathlib.Path): Path where to save the models.

    Returns:
        None

    """

    def to_parallelize(spec, model, ordered_features):
        # prepare data
        n_features = spec.data_kwargs["n_features"]
        n_obs = spec.data_kwargs["n_obs"]
        order_features = bool(spec.data_kwargs["order_features"])

        X, y = load_data(model, n_train=n_obs)
        XX = subset_features(X, order_features, ordered_features, n_features)

        # fit and save model
        model = surrogates.fit(model_type=spec.model, X=XX, y=y, **spec.model_kwargs)
        model_path = save_path / spec.identifier
        surrogates.save(model, model_path, overwrite=True)

        # status report
        print(f"Done: {spec.identifier}")

    with parallel_backend("threading", n_jobs=4):
        Parallel()(
            delayed(to_parallelize)(spec, model, ordered_features)
            for spec in specifications
        )


@click.command()
@click.argument("model", type=str)
def main(model):
    with open(ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl"), "rb") as f:
        specifications = pickle.load(f)
    try:
        ordered_features = pd.read_csv(ppj("OUT_DATA", f"{model}-ordered_features.csv"))
    except FileNotFoundError:
        ordered_features = None

    save_path = Path(ppj("OUT_FITTED_MODELS")) / model
    save_path.mkdir(parents=True, exist_ok=True)

    _fit(model, specifications, ordered_features, save_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
