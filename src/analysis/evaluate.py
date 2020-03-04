"""Evaluate fitted models by using the predicted outcomes."""
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

from bld.project_paths import project_paths_join as ppj
from src.analysis.auxiliary import compute_loss_given_metrics
from src.auxiliary.auxiliary import load_implemented_models
from src.data_management.auxiliary import load_testing_data

if __name__ == "__main__":
    metrics = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "mdae": median_absolute_error,
    }

    surrogates = load_implemented_models()
    X, y, _ = load_testing_data()

    predictions = pickle.load(open(ppj("OUT_DATA", "predictions.pkl"), "rb"))

    losses = {}
    for surrogate in surrogates:
        prediction = predictions[[surrogate]].values
        loss = compute_loss_given_metrics(y, prediction, metrics)
        losses[surrogate] = loss

    df = pd.DataFrame(losses)
    tidy = (
        df.reset_index(level=0)
        .rename(columns={"index": "measure"})
        .melt(id_vars="measure", var_name="method", value_name="loss")
    )
    tidy.to_csv(ppj("OUT_DATA", "losses.csv"), index=False)
