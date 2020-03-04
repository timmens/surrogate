"""Predict outcomes of test set using the fitted models and save predictions."""
import pickle

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.analysis.auxiliary import get_surrogate_instances
from src.auxiliary.auxiliary import load_implemented_models
from src.data_management.auxiliary import load_testing_data

if __name__ == "__main__":

    surrogates = load_implemented_models()
    surrogate_classes = get_surrogate_instances(surrogates)

    X, _, _ = load_testing_data()
    predictions = []
    for i, surrogate in enumerate(surrogate_classes):
        surrogate.load(ppj("OUT_ANALYSIS", surrogates[i] + ".csv"))
        prediction = surrogate.predict(X)
        predictions.append(prediction.reshape((-1, 1)))

    prediction_array = np.concatenate(predictions, axis=1)
    out = pd.DataFrame(prediction_array, columns=surrogates)

    pickle.dump(out, open(ppj("OUT_DATA", "predictions.pkl"), "wb"))
