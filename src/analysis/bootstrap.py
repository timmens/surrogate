"""Bootstrap analysis of mae of mdoels for varying number of observations."""
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from sklearn.metrics import mean_absolute_error

from bld.project_paths import project_paths_join as ppj
from src.model_code import PolynomialRegression
from src.model_code import RidgeRegression
from src.utilities.utilities import load_testing_data
from src.utilities.utilities import load_training_data


def _bootstrap(n_samples, n_obs_list, models_params, metric, n_jobs, seed):
    """Wrapper for ``_bootstrap_single_model`` applied to multiple models.

    Args:
        n_samples (int): Number of bootstrap draws for each number of observations.
        n_obs_list (list): List containing the number of observations.
            Example. nobs = [100, 500, 1000]
        models_params (dict): Dictionary of dictionaries representing the different
            models. Each inner dictionary has to contain the entry ``model``, which must
            be a model from ``src.model_code`` and ``fit_kwargs``, which must denote the
            kwargs used for fitting ``model``. Example.
            models_params = {
                "polreg": {
                    "model": polynomialregression,
                    "fit_kwargs": {"fit_intercept": True, "degree": 2}
                }
            }
        metric (function): A metric from ``sklearn.metrics``.
        n_jobs (int): Number of jobs for parallelization.
        seed (int): Random number seed.

    Returns:

    """
    out = {}
    for key, params in models_params.items():
        result = _bootstrap_single_model(
            n_samples=n_samples,
            n_obs_list=n_obs_list,
            model=params["model"],
            fit_kwargs=params["fit_kwargs"],
            metric=metric,
            n_jobs=n_jobs,
            seed=seed,
        )
        out[key] = result

    return out


def _bootstrap_single_model(
    n_samples, n_obs_list, model, fit_kwargs, metric, n_jobs, seed
):
    """Compute metric for model for varying sample size.

    Compute the mean absolute error on the test sample for varying
    number observation. For each number of observation draw
    ``nsamples`` bootstrap samples.

    Args:
        n_samples (int): Number of bootstrap draws for each number of observations.
        n_obs_list (list): List containing the number of observations.
            Example. nobs = [100, 500, 1000]
        model (Model object): A model from ``src.model_code``.
        fit_kwargs (dict): Parameters used in ``fit`` method of ``model``.
        metric (function): A metric from ``sklearn.metrics``.
        n_jobs (int): Number of jobs for parallelization.
        seed (int): Random number seed.

    Returns:
        df_tidy (pd.DataFrame): Tidy data frame with columns "n" and "mae".

    """
    Xtest, ytest = load_testing_data()
    X, y = load_training_data()

    def to_parallelize(k):
        """Wrapper function over which we parallelize."""
        result = np.empty(len(n_obs_list))
        for i, n in enumerate(n_obs_list):
            index = _compute_subset_index(max_length=X.shape[0], size=n, seed=seed + k)
            XX, yy = X.iloc[index, :], y[index]

            m = model()
            # check for dimensionality problem; see issue #6
            if m.name == "PolynomialRegression" and m.degree == 2 and XX.shape[0] < 406:
                result[i] = np.nan
            else:
                m.fit(XX, yy, **fit_kwargs)
                ypred = m.predict(Xtest)
                loss = metric(ytest, ypred)
                result[i] = loss

        return result

    data = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(to_parallelize)(i) for i in range(n_samples)
    )
    data_tabular = np.stack(data)

    df = pd.DataFrame(data_tabular, columns=n_obs_list)

    df_tidy = df.melt(var_name="n", value_name="mae")
    return df_tidy


def _compute_subset_index(max_length, size, seed):
    """Sample elements from index with replacement.

    Sample ``size`` many elements from {1,...,``max_length``} with replacement and
    return the resulting index as a pd.Index.

    Args:
        max_length (int): Length of index.
        size (int): Size of subindex.
        seed (int): Random number seed.

    Returns:
        indices (list): List containing indices to subset the
            training data set.

    """
    np.random.seed(seed)

    index = pd.Index(
        np.random.choice(range(max_length), size=size, replace=False)
    ).sort_values()
    return index


if __name__ == "__main__":
    # models we want to evaulate using the (some) bootstrap
    models = {
        "linreg": {
            "model": PolynomialRegression,
            "fit_kwargs": {"degree": 1, "fit_intercept": True},
        },
        "polreg": {
            "model": PolynomialRegression,
            "fit_kwargs": {"degree": 2, "fit_intercept": True},
        },
        "ridgereg": {
            "model": RidgeRegression,
            "fit_kwargs": {"degree": 2, "fit_intercept": True},
        },
    }

    # data parameters
    n_obs = [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        2000,
        5000,
        7000,
        10000,
    ]

    data = _bootstrap(
        n_samples=20,
        n_obs_list=n_obs,
        models_params=models,
        metric=mean_absolute_error,
        n_jobs=4,
        seed=1,
    )
    df = pd.concat(data, axis=0)

    df_tidy = df.reset_index(level=0).rename(columns={"level_0": "model"})
    df_tidy.to_csv(ppj("OUT_ANALYSIS", "bootstrap_mae.csv"), index=False)
