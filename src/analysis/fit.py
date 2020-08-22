"""Load specific model and fit to training data set."""
from pathlib import Path

import click
import cloudpickle
from tqdm import tqdm

import src.surrogates as surrogates
from bld.project_paths import project_paths_join as ppj
from src.specs import Specification  # noqa: F401
from src.utilities import load_data


def _fit(simulation_model, specifications, save_path):
    """Fit all models specified in ``specifications`` and save to disc.

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

    def to_run(spec):
        X, y = load_data(simulation_model, n_train=spec.n_obs)
        predictor = surrogates.fit(spec.model, X, y, **spec.fit_kwargs)
        surrogates.save(predictor, save_path / spec.identifier, overwrite=True)

    for spec in tqdm(specifications):
        to_run(spec)


@click.command()
@click.argument("model", type=str)
def main(model):
    with open(ppj("OUT_MODEL_SPECS", f"{model}-specifications.pkl"), "rb") as f:
        specifications = cloudpickle.load(f)

    save_path = Path(ppj("OUT_FITTED_MODELS")) / model
    save_path.mkdir(parents=True, exist_ok=True)

    _fit(model, specifications, save_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
