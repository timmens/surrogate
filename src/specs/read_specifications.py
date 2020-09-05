import yaml

from src.config import SRC


def gather_spec_files():
    """Gather all specification files from path ``src/specs``."""
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


def filter_specification_that_are_run(specifications):
    """Remove specifications that have key "run" set to False and remove key "run"."""
    specifications = {
        name: spec for name, spec in specifications.items() if spec["run"]
    }
    for _, spec in specifications.items():
        spec.pop("run")
    return specifications


def remove_xscale_key(specifications):
    specifications = specifications.copy()
    for _, spec in specifications.items():
        spec.pop("xscale", None)
    return specifications


def read_specifications(fitting=True):
    """Read specification files of projects that shall be run."""
    files = gather_spec_files()
    specifications = read_files(files)
    specifications = filter_specification_that_are_run(specifications)
    if fitting:
        specifications = remove_xscale_key(specifications)
    return specifications
