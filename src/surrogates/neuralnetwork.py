"""Function wrapper to use neural networks for regression"""
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler


def fit(X, y, layers, n_epochs=100, n_batch_size=20):
    """Fit a neural network regressor.

    Args:
        X (pd.DataFrame): Data on features.
        y (pd.Series or np.ndarray): Data on outcomes.
        layers (str): str specifying the number of hidden layers and hidden nodes in the
            neural network. Example: "100-100-100".
        n_epochs (int): Number of epochs used for model fitting.
        n_batch_size (int): Batch size used for model fitting.

    Returns:
        predictor (dict): Dictionary with entries 'model' for the fitted model and
            'pipe' for the pre-processing pipeline.
            - model : keras.wrappers.scikit_learn.KerasRegressor
            - pipe : sklearn.pipeline.Pipeline

    """
    pipe = StandardScaler()
    XX = pipe.fit_transform(X)

    build_regressor = _get_build_regressor_func(input_dim=XX.shape[1], layers=layers)

    nnet = KerasRegressor(
        build_fn=build_regressor, batch_size=n_batch_size, epochs=n_epochs
    )
    nnet.fit(XX, y, verbose=False)

    predictor = {"model": nnet, "pipe": pipe}
    return predictor


def predict(X, predictor):
    """Predict outcome using the fitted model and new data.

    Args:
        X (pd.DataFrame): New data on features.
        predictor (namedtuple): Named tuple with entries 'model' for the fitted model
            and 'pipe' for the pre-processing pipeline.
                model : keras.wrappers.scikit_learn.KerasRegressor
                pipe : sklearn.pipeline.Pipeline

    Returns:
        predictions (np.array): The predicted outcomes.

    """
    XX = predictor["pipe"].transform(X)
    predictions = predictor["model"].predict(XX)
    return predictions


def _get_build_regressor_func(input_dim, layers):
    """Create a function to build a neural network regressor.

    The ``build_regressor`` function is needed by the Keras model to build the neural
    network. Here we create this function dynamically.

    Example for layers. ``layers = "54-81-54"`` means that the first hidden layer has
    54 nodes, the second hidden layer has 81 nodes, the third hidden layer has again 54
    nodes and implicitly the output layer has one node.

    Args:
        input_dim (int): Number of features.
        layers (str): String describing the number of nodes per hidden layer.

    Returns:
        build_regressor (function): Function to build a neural net regressor.

    """
    layers = [int(n) for n in layers.split("-")]

    def build_regressor():
        regressor = Sequential()
        regressor.add(Dense(units=layers[0], activation="relu", input_dim=input_dim))
        for u in layers[1:]:
            regressor.add(Dense(units=u, activation="relu"))
        regressor.add(Dense(units=1, activation="linear"))
        regressor.compile(optimizer="adam", loss="mean_absolute_error")
        return regressor

    return build_regressor
