"""Create data for special plots."""
from itertools import count

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from bld.project_paths import project_paths_join as ppj
from src.data_management.utilities import compute_testing_loss
from src.data_management.utilities import load_testing_data
from src.data_management.utilities import load_training_data
from src.model_code.polynomialregression import PolynomialRegression
from src.model_code.ridgeregression import RidgeRegression


def bootstrap_metric_varying_sample_size(nobs, nsamples, model, metric, seed):
    """Compute metric for model for varying sample size.

    Compute the mean absolute error on the test sample for varying
    number observation. For each number of observation draw
    ``nsamples`` bootstrap samples.

    Args:
        nobs (list): List containing the number of observations.
            Example. nobs = [100, 500, 1000]
        nsamples (int): Number of bootstrap draws for each number
            of observations.
        model (Model object): A model from ``src.model_code``.
        metric (function): A metric from ``sklearn.metrics``.
        seed (int): Random number seed.

    Returns:
        df_tidy (pd.DataFrame): Tidy data frame with columns
            "n" and "mae".

    """
    counter = count(seed)

    Xtest, ytest = load_testing_data()

    curves = np.empty((nsamples, len(nobs)))
    for i in range(nsamples):
        for j, n in enumerate(nobs):
            X, y = load_training_data(nobs=n, seed=next(counter))
            model.fit(X, y, degree=2)
            error = compute_testing_loss(
                model=model, ytest=ytest, Xtest=Xtest, measure=metric
            )
            curves[i, j] = error

    df = pd.DataFrame(curves, columns=nobs)

    df_tidy = df.melt(var_name="n", value_name="mae")
    return df_tidy


if __name__ == "__main__":
    NOBS = [500, 1000, 2500, 5000, 10000]
    NSAMPLES = 25

    pr = PolynomialRegression()
    rr = RidgeRegression()

    df_mae_polynomial = bootstrap_metric_varying_sample_size(
        nobs=NOBS, nsamples=NSAMPLES, model=pr, metric=mean_absolute_error, seed=1
    )

    df_mae_ridge = bootstrap_metric_varying_sample_size(
        nobs=NOBS, nsamples=NSAMPLES, model=rr, metric=mean_absolute_error, seed=1
    )

    df_mae_polynomial.to_csv(ppj("OUT_ANALYSIS", "bootstrap_mae_polynomial.csv"))
    df_mae_ridge.to_csv(ppj("OUT_ANALYSIS", "bootstrap_mae_ridge.csv"))
