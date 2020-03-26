"""Predict outcomes of test set using the fitted models and save predictions."""
import pickle

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.auxiliary.auxiliary import get_surrogate_instances
from src.auxiliary.auxiliary import load_sorted_features
from src.auxiliary.auxiliary import load_surrogates_specs
from src.data_management.utilities import load_testing_data

if __name__ == "__main__":

    # load dictionary of fitted model specifications
    specs = load_surrogates_specs()

    # extract model keys
    keys = list(specs.keys())

    # extract surrogate model name
    names = [specs[key]["model"] for key in keys]

    # initiate model classes
    classes = get_surrogate_instances(names)

    X, _, = load_testing_data()
    predictions = []

    # load order of importance of features
    ordered_features = load_sorted_features()

    for key, model in zip(keys, classes):
        model.load(ppj("OUT_ANALYSIS", key))

        features = specs[key]["features"]
        features = features if features != 0 else len(ordered_features)
        feature_list = ordered_features[:features]

        prediction = model.predict(X[feature_list])
        predictions.append(prediction.reshape((-1, 1)))

    prediction_array = np.concatenate(predictions, axis=1)
    out = pd.DataFrame(prediction_array, columns=keys)

    pickle.dump(out, open(ppj("OUT_DATA", "predictions.pkl"), "wb"))
