"""Helper functions for scripts in ``src.analysis``."""
import importlib

from src.auxiliary.auxiliary import get_model_class_names


def get_surrogate_instances(surrogates):
    """

    Args:
        surrogates:

    Returns:

    """
    module_names = ["src.model_code." + surrogate for surrogate in surrogates]
    modules = [importlib.import_module(module) for module in module_names]
    surrogate_class_names = get_model_class_names(surrogates)
    surrogate_classes = [
        getattr(modules[i], surrogate_class_names[i])() for i in range(len(surrogates))
    ]

    return surrogate_classes
