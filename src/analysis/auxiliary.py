"""Helper functions for scripts in ``src.analysis``."""
import importlib

import pandas as pd

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


def compute_loss_given_metrics(ytrue, ypredicted, metrics):
    """Compute loss of prediction given various metrics.

    Args:
        ytrue (np.array): True outcome.
        ypredicted (np.array): Predicted outcome.
        metrics (dict): Dictionary containing metrics.

    Returns:
        loss (pd.Series): The occured loss from prediction.

    """
    loss = []
    for _, metric in metrics.items():
        loss.append(metric(ytrue, ypredicted))

    loss = pd.Series(loss, index=metrics.keys())
    return loss
