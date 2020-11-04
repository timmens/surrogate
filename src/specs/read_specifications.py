import yaml

from src.config import SRC


def read_specifications(fitting=True):
    """Read specification files of projects that shall be run.

    Args:
        fitting (bool): Are specifications read to fit models or not.

    Returns:
        specifications (dict): Dictionary containing specifications. Keys are named of
            specification files, values are again dictionaries with specified values.

    """
    files = gather_specification_files()
    specifications = read_files(files)
    specifications = drop_specifications_that_are_not_run(specifications)
    if fitting:
        specifications = remove_plotting_keys(specifications)
    return specifications


def drop_specifications_that_are_not_run(specifications):
    """Drop specification that should not be run.

    Removes specifications that have key "run" set to False and then removes key "run".

    Args:
        specifications (dict): Dictionary containing specifications. Keys are named of
            specification files, values are again dictionaries with specified values.

    Returns:
        specifications (dict): As above but without without specifications that had
            key "run" to true and without key "run".

    """
    specifications = {
        name: spec for name, spec in specifications.items() if spec["run"]
    }
    for _, spec in specifications.items():
        spec.pop("run")
    return specifications


def gather_specification_files():
    """Gather specification files.

    Gathers all yaml files from folder ``src/specs`` and returns them in a list.py

    Returns:
        files (list): List of paths to yaml specification files.

    """
    p = SRC / "specs"
    files = list(p.glob("*.yaml"))
    return files


def read_files(files):
    """Read specifications.

    Args:
        files (list): List of paths to yaml specification files.

    Returns:
        specifications (dict): Dictionary containing specifications. Keys are named of
            specification files, values are again dictionaries with specified values.

    """
    specifications = {f.stem: yaml.safe_load(open(f, "r")) for f in files}
    return specifications


def remove_plotting_keys(specifications):
    """Remove plotting keys if set."""
    specifications = specifications.copy()
    for _, spec in specifications.items():
        spec.pop("xscale", None)
        spec.pop("xticks", None)
    return specifications
