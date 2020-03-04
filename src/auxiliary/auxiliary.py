"""Helper functions."""
import json
import pathlib
import warnings

from bld.project_paths import project_paths_join as ppj


def load_implemented_models(ignore=("__init__", "surrogate")):
    """Load implemented models.

    Load models implemented in directory ``src.model_code`` and return model names as
    a list while not return models specified in the argument *ignore*.

    Args:
        ignore (tuple or list): Which files to ignore. Defaults to ('__init__',
            'surrogate').

    Returns:
        files (list): List of names of implemented models.

    """
    p = pathlib.Path(ppj("IN_MODEL_CODE", ""))
    files = [str(file) for file in p.glob("*.py")]

    if len(files) == 0:
        warnings.warn("No models implemented.", UserWarning)
        return None

    files = [_extract_filename_from_path(file) for file in files]
    files = [file for file in files if file not in ignore]
    return files


def _extract_filename_from_path(file):
    """Extract name of file from complete directory path.

    Args:
        file (str): File path.

    Returns:
        f (str): The name of the file.
    """
    f = file.split("/")[-1]
    f = f.split(".")[0]
    return f


def get_model_class_names(models):
    """Translate model names to names of respective classes.

    Args:
        models (list): List of model names.

    Returns:
        translated (list): List model class names.
    """
    translation = {
        "linearregression": "LinearRegression",
        "polynomialregression": "PolynomialRegression",
    }

    translated = [translation[model] for model in models]
    return translated


def load_surrogate_fit_params():
    """Load model specifications for ``fit`` method of surrogates.

    Returns:
        out (dict): Dictionary containing parameters for ``fit`` method of surrogates.

    Example:
    >>>params = load_surrogate_fit_params()
    >>>params['PolynomialRegression']
    {"degree":  2, "fit_intercept":  true}

    """
    with open(ppj("IN_MODEL_SPECS", "surrogate_fit_params.json")) as file:
        out = json.loads(file.read())
    return out
