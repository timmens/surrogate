"""Variable selection using ridge regression"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import src.surrogates as surrogate
from bld.project_paths import project_paths_join as ppj
from src.utilities.utilities import load_data


def is_interaction(coef_name):
    return ":" in coef_name


def is_squared(coef_name):
    return coef_name.endswith("^2")


def main():
    # data used for ridge regression (low sample size to induce regularization)
    Xridge, yridge = load_data(n_train=200)

    # data used for polynomial model fitting on subset of variables
    Xpol, ypol = load_data(n_train=5000)

    # data used to estimate the mean absolute error on test set
    Xtest, ytest = load_data(n_test=5000, testing=True)

    # ridge regression, variable regularization
    rr = surrogate.fit("ridge", Xridge, yridge, degree=1)

    coef = rr["model"].coefficients.values.reshape(-1)
    thresholds = np.linspace(0, 0.05, num=500)

    # find parameters which are zero given a threshold
    is_zero = []
    for thresh in thresholds:
        zero = np.where(np.abs(coef) < thresh)[0]
        is_zero.append(zero)

    # extract parameter names
    is_zero_named = [
        rr["model"].coefficients.index[index].to_list() for index in is_zero
    ]

    is_zero_squared = [  # noqa: F841
        [e for e in x if is_squared(e)] for x in is_zero_named
    ]
    is_zero_interaction = [  # noqa: F841
        [e for e in x if is_interaction(e)] for x in is_zero_named
    ]
    is_zero_linear = [
        [e for e in x if not is_interaction(e) and not is_squared(e)]
        for x in is_zero_named
    ]

    # compute test mae using polynomial model and store in data frame
    mae = []
    for drop in is_zero_named:
        XX = Xpol.drop(drop, axis=1)  # noqa: F841
        XXtest = Xtest.drop(drop, axis=1)
        pr = surrogate.fit("polynomial", XXtest, ytest, degree=2, fit_intercept=True)
        ypred = pr["model"].predict("polynomial", XXtest)
        mae.append(mean_absolute_error(ytest, ypred))

    df = pd.DataFrame(zip(mae, thresholds), columns=["mae", "thresholds"])

    # compute when the set of variables that are set to zero changes
    changes = []
    change_index = []
    for i in range(len(is_zero_linear) - 1):
        e = set(is_zero_linear[i])
        ee = set(is_zero_linear[i + 1])
        if e != ee:
            change_index.append(i + 1)
            changes.append(list(ee - e)[0])

    # save data
    data = {
        "df": df,
        "change_index": change_index,
        "changes": changes,
        "thresholds": thresholds,
    }
    with open(ppj("OUT_ANALYSIS", "variable_selection.pkl"), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
