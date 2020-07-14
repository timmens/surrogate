"""Load specific model and fit to training data set."""
import pickle
from pathlib import Path

import click
from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend
from tqdm import tqdm

import src.surrogates as surrogates
from bld.project_paths import project_paths_join as ppj
from src.specs import Specification  # noqa: F401
from src.utilities.utilities import load_data


def _fit(simulation_model, specifications, save_path):
    """Fit all models specified in ``specifications``.

    Args:
        simulation_model (str): (Economic) simulation model. Must be in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended'].
        specifications (list): List of namedtuple model specifcations with entries
            'model', 'identifier', 'fit_kwargs', 'predict_kwargs', corresponding to the
            specifications for ``model``.
        save_path (pathlib.Path): Path where to save the models.

    Returns:
        None

    """

    def to_parallelize(spec, simulation_model=simulation_model, save_path=save_path):
        # prepare training data
        X, y = load_data(simulation_model, n_train=spec.n_obs)
        # fit
        predictor = surrogates.fit(model_type=spec.model, X=X, y=y, **spec.fit_kwargs)
        # save
        surrogates.save(predictor, save_path / spec.identifier, overwrite=True)

    with parallel_backend("threading", n_jobs=4):
        Parallel()(delayed(to_parallelize)(spec) for spec in tqdm(specifications))


@click.command()
@click.argument("model", type=str)
def main(model):
    with open(ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl"), "rb") as f:
        specifications = pickle.load(f)

    save_path = Path(ppj("OUT_FITTED_MODELS")) / model
    save_path.mkdir(parents=True, exist_ok=True)

    _fit(model, specifications, save_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
