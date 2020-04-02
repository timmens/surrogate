"""Evaluate fitted models by using the predicted outcomes."""
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

from bld.project_paths import project_paths_join as ppj
from src.analysis.auxiliary import compute_loss_given_metrics
from src.auxiliary.auxiliary import load_surrogates_specs
from src.data_management.utilities import load_testing_data

if __name__ == "__main__":
    metrics = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "mdae": median_absolute_error,
    }

    # load dictionary of model specifications
    specs = load_surrogates_specs()

    # extract model keys
    keys = list(specs.keys())

    # extract model names
    names = [specs[key]["model"] for key in specs.keys()]

    X, y = load_testing_data()

    predictions = pickle.load(open(ppj("OUT_ANALYSIS", "predictions.pkl"), "rb"))

    losses = {}
    for key in keys:
        prediction = predictions[[key]].values
        loss = compute_loss_given_metrics(y, prediction, metrics)
        losses[key] = loss

    df = pd.DataFrame(losses)
    tidy = (
        df.reset_index(level=0)
        .rename(columns={"index": "measure"})
        .melt(id_vars="measure", var_name="method", value_name="loss")
    )
    tidy.to_csv(ppj("OUT_ANALYSIS", "losses.csv"), index=False)
