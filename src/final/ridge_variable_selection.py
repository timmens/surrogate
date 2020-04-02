"""Analyze and plot variable selection via ridge regression.

Content should be split into plot and data analysis.
"""
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from bld.project_paths import project_paths_join as ppj
from src.data_management.utilities import load_testing_data
from src.data_management.utilities import load_training_data
from src.model_code.polynomialregression import PolynomialRegression
from src.model_code.ridgeregression import RidgeRegression


def is_interaction(coef_name):
    return ":" in coef_name


def is_squared(coef_name):
    return coef_name.endswith("^2")


if __name__ == "__main__":
    # data used for ridge regression (low sample size to induce regularization)
    Xridge, yridge = load_training_data(nobs=200)
    # data used for polynomial model fitting on subset of variables
    Xpol, ypol = load_training_data(nobs=5000)
    # data used to estimate the mean absolute error on test set
    Xtest, ytest = load_testing_data(nobs=5000)

    rr = RidgeRegression()
    pr = PolynomialRegression()

    # ridge regression, variable regularization
    rr = rr.fit(Xridge, yridge, degree=1)

    coef = rr.coefficients.values.reshape(-1)

    thresholds = np.linspace(0, 0.05, num=500)

    # find parameters which are zero given a threshold
    is_zero = []
    for thresh in thresholds:
        zero = np.where(np.abs(coef) < thresh)[0]
        is_zero.append(zero)

    # extract parameter names
    is_zero_named = [rr.coefficients.index[index].to_list() for index in is_zero]

    is_zero_squared = [[e for e in x if is_squared(e)] for x in is_zero_named]
    is_zero_interaction = [[e for e in x if is_interaction(e)] for x in is_zero_named]
    is_zero_linear = [
        [e for e in x if not is_interaction(e) and not is_squared(e)]
        for x in is_zero_named
    ]

    # compute test mae using polynomial model and store in data frame
    mae = []
    for drop in is_zero_named:
        XX = Xpol.drop(drop, axis=1)
        XXtest = Xtest.drop(drop, axis=1)
        pr = pr.fit(XX, ypol, degree=2, fit_intercept=True)
        ypred = pr.predict(XXtest)
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

    # the plot
    sns.set(style="ticks", rc={"lines.linewidth": 3}, font_scale=1.2)
    fig, ax = plt.subplots()

    fig.set_size_inches(13, 8.27)
    ax.set(xlim=(0, 0.023))
    ax.set(ylim=(0.005, 0.025))

    np.random.seed(4)
    sns.lineplot(x="thresholds", y="mae", data=df)
    i = 0
    for change, change_text in zip(change_index, changes):
        i += 1
        if i > 15:
            break
        add = -0.001 * i
        change_thresh = thresholds[change]
        plt.axvline(x=change_thresh, color="r", linewidth=0.5)
        plt.annotate(xy=(change_thresh, 0.02 + add), s=change_text, weight="bold")
    plt.title("Variable selection using Ridge Regression")
    plt.xlabel("Coefficient threshold")
    plt.ylabel("Mean abs. error on test set")
    plt.annotate(
        xy=(0.0005, 0.0205),
        s="Blue line shows the mean absolute error when using\na "
        "newly fit 2nd degree polynomial that only\nuses "
        "variables which had a coefficient value\nabove the "
        "threshold. Annotated red lines display\nwhen and which "
        "variable was dropped. (Variables\n included were"
        "standardized)",
        bbox={"facecolor": "white", "edgecolor": "gray", "boxstyle": "round"},
    )

    fig_path = ppj("OUT_FIGURES", "ridge_variable_selection.pdf")
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.7)

    # move figure to sciebo
    sciebo_path = "/home/tm/sciebo/uni-master/master-thesis/structUncertainty/"
    shutil.copy(fig_path, sciebo_path + "ridge_variable_selection.pdf")
