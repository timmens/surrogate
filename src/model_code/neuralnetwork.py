"""Neural network regression surrogate model."""
import glob
import os
import pickle
import warnings

from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

from src.model_code.surrogate import assert_input_fit
from src.model_code.surrogate import Surrogate
from src.utilities import suppress_stdout


class NeuralNetwork(Surrogate):
    """Neural network regression surrogate model."""

    def __init__(self):
        self.nnet = None
        self.scaler = None
        self.fit_history = None
        self.is_fitted = False
        self.file_type = ".h5"
        super().__init__()

    def fit(self, X, y, **kwargs):
        """Fit a neural network regression model.

       Fits a neural network regresion on. The architecture is specified by ``kwargs``.

        Args:
            X (pd.DataFrame):
                Data on features.

            y (pd.Series or np.ndarray):
                Data on outcomes.

            **kwargs:
                - layers (list or tuple): List or tuple specifying the number of hidden
                    layers and how many nodes each layer has. Example. layers =
                    [54, 81, 54] means that the input dimension the first hidden layer
                    has 54 nodes, the second hidden layer has 81 nodes, the third hidden
                    layer has again 54 nodes and implicitly the output layer has one
                    node.

                - n_epochs (int): Number of epochs used for model fitting.

                - n_batch_size (int): Batch size used for model fitting.

        Returns:
            self: The fitted neural network object.

        """
        result = _fit(X=X, y=y, **kwargs)

        self.nnet = result["nnet"]
        self.scaler = result["scaler"]
        self.fit_history = result["fit_history"]
        self.is_fitted = True

        return self

    def predict(self, X):
        """Predict outcome using the fitted model and new data.

        Uses the fitted coefficients in ``self.coefficients`` and new data
        ``X`` to predict outcomes. If the model has not been fitted yet we
        return None and a warning.

        Args:
            X (pd.DataFrame): New data on features (unprocessed).

        Returns:
            - None, if ``self.is_fitted`` is False and otherwise
            - predictions (np.array): The predicted outcomes.

        """
        if not self.is_fitted:
            warnings.warn("The model has not been fitted yet.", UserWarning)
            return None

        predictions = _predict(X=X, scaler=self.scaler, nnet=self.nnet)
        return predictions

    def save(self, filename, overwrite=False):
        """Save fitted model to disc.

        Save the fitted coefficents stored under ``self.coefficients`` to disc
        as a csv file. Can be loaded afterwards using the method ``load``.

        Args:
            filename (str): File path except file type ending. If file type ending is
                provided it will be ignored.
            overwrite (bool): Should the file be overwritten if it exists.

        Returns:
            None

        """
        file_path, _ = os.path.splitext(filename)
        _save(
            nnet=self.nnet,
            scaler=self.scaler,
            file_path=file_path,
            format=self.file_type,
            overwrite=overwrite,
        )

    def load(self, filename):
        """Load a fitted model from disc.

        Load fitted coefficients stored as a csv from disc and store them under
        ``self.coefficients``. Will work on files created by method ``save``,
        but also files manually created with first named column "coefficient"
        and second column "value".

        Args:
            filename (str): File path except file type ending. If file type ending is
                provided it will be ignored.

        Returns:
            None

        """
        file_path, file_type = os.path.splitext(filename)
        model = _load(file_path, self.file_type)

        self.scaler = model["scaler"]
        self.nnet = model["nnet"]
        self.is_fitted = True

        return self

    @property
    def name(self):
        """Return name of class as string."""
        return self.__class__.__name__


def _fit(X, y, layers, n_epochs=100, n_batch_size=10):
    """Fit a ridge regression model on polynomial features.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        layers (list or tuple): List or tuple specifying the number of hidden layers and
            hidden nodes in the neural network.
        n_epochs (int): Number of epochs used for model fitting.
        n_batch_size (int): Batch size used for model fitting.

    Returns:
        out (dict): Output dictionary containing:
            - nnet (KerasRegressor): Fitted neural network.
            - scaler (sklearn.preprocessing.StandardScaler): Fitted scaler.

    """
    assert_input_fit(X=X, y=y)

    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X_ = scaler.transform(X)

    architecture = {
        "input_dim": X_.shape[1],
        "layers": layers,
    }
    build_regressor = create_build_regressor_func(architecture)

    nnet = KerasRegressor(
        build_fn=build_regressor, batch_size=n_batch_size, epochs=n_epochs
    )

    with suppress_stdout():
        fit_history = nnet.fit(X_, y)

    out = {
        "nnet": nnet,
        "scaler": scaler,
        "fit_history": fit_history,
    }
    return out


def _predict(X, scaler, nnet):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used to fit.
        nnet (KerasRegressor): Fitted neural network.

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    X_ = scaler.transform(X)

    y_pred = nnet.predict(X_)
    return y_pred


def _save(nnet, scaler, file_path, format, overwrite):
    """Save fitted model to disc.

    Save the fitted coefficents to disc as a csv file. Since the number of degrees of
    the polynomial is important we create a new row with name "<--degree-->" and value
    ``degree``. We choose the weird name so that the chance is low that a standard
    feature name is equal to it.

    Args:
        nnet (KerasRegressor): Fitted neural network.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used to fit.
        file_path (str): File path.
        format (str): File format of saved model coefficients.
        overwrite (bool): Should the file be overwritten if it exists.

    Returns:
        None

    """
    file_present = glob.glob(file_path + format)
    if overwrite or not file_present:
        # save model
        nnet.model.save(file_path + format)

        # save scaler
        with open(file_path + "_scaler.pkl", "wb") as handle:
            pickle.dump(scaler, handle)
    else:
        warnings.warn("File already exists. No actions taken.", UserWarning)


def _load(file_path, format):
    """Load a fitted model from disc.

     Load fitted coefficients stored as a csv from disc and return them. Will
     work on files created by method ``save``, but also files manually created
     with first named column "coefficient", which will be set as the index.
     For correct functionality of the other functions the second column needs
     to be called "value". If the file to load was not created using the ``save``
     method one has to manually add a row with name "<--degree-->" and value equal to
     the degree of the polynomial which correspond to the coefficients.

    Args:
        file_path (str): File path, without file format.
        format (str): Format of how model is saved. Example: format=".csv".

    Returns:
        out (dict): Dictionary containing
            - coefficients (pd.DataFrame): The named coefficient values.
            - scaler (sklearn.preprocessing.StandardScaler): Fitted scaler.
            - degree (int): The degree of the polynomial.

    Raises:
        ValueError, if index column name is wrongly specified in csv file.
        AssertionError, if value column name is incorrect.

    """
    try:
        # load scaler
        with open(file_path + "_scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)

        # load model
        nnet = load_model(file_path + format)
    except Exception:
        raise ValueError(
            f"Model cannot be loaded. Is there a file called"
            f"{file_path + format} and {file_path}_scaler.pkl?"
        )

    out = {
        "scaler": scaler,
        "nnet": nnet,
    }
    return out


def create_build_regressor_func(architecture):
    """Create a function to build a neural network regressor.

    The ``build_regressor`` function is needed by the Keras model to build the neural
    network. Here we create this function dynamically.

    Args:
        architecture (dict): Architecture of the neural network regressor.
            Example. architecture = {'input_dim': 27, 'layers': [54, 81, 54]}, which
            means that the input dimension (number of features) is 27, the first hidden
            layer has 54 nodes, the second hidden layer has 81 nodes, the third hidden
            layer has again 54 nodes and implicitly the output layer has one node.

    Returns:
        build_regressor (function): Function to build a neural net regressor.

    """
    input_dim = architecture["input_dim"]
    layers = architecture["layers"]

    func_layers = [
        f"regressor.add(Dense(units={u}, activation='relu'))" for u in layers[1:]
    ]
    func_layers_code = ";".join(func_layers)

    def build_regressor():
        regressor = Sequential()
        regressor.add(Dense(units=layers[0], activation="relu", input_dim=input_dim))
        exec(func_layers_code)
        regressor.add(Dense(units=1, activation="linear"))
        regressor.compile(optimizer="adam", loss="mean_absolute_error")
        return regressor

    return build_regressor
