"""Generic surrogate model class."""
from abc import ABC
from abc import abstractmethod


class Surrogate(ABC):
    """Generic surrogate model class."""

    def __init__(self):
        """Initiate surrogate object."""
        super().__init__()

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Generic fit method.

        Fit the surrogate model on the ``data`` using the ``formula``.

        Args:
            X (pd.DataFrame): Data on features.
            y (pd.Series or np.ndarray): Data on outcomes.
            **kwargs: Keyword arguments specific for the individual surrogate.

        Returns:
            self: The fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Generic predict method.

        Predict outcomes using new ``data`` and the fitted model.

        Args:
            X (pd.DataFrame): New data on features.

        Returns:
            predictions (np.array): The predictions using the fitted surrogate
                model und new ``data``.
        """
        pass

    @abstractmethod
    def save(self, filename, overwrite):
        """Generic method to save the fitted model.

        Args:
            filename (str): File path.
            overwrite (bool): Should the file be overwritten if it exists.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, filename):
        """Generic method to load a fitted model.

        Args:
            filename (str): File path.

        Returns:
            self: The fitted estimator.
        """
        pass


def assert_input_fit(X, y):
    """

    Args:
        X:
        y:

    Returns:

    """
    return True
