"""Helper functions for scripts in ``src.analysis``."""
import pandas as pd


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
