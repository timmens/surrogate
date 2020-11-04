import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.surrogates as surrogate
from src.config import BLD
from src.config import SRC
from src.specs import read_specifications


def uncertainty_propagation_plots_surrogates(benchmark, predictions):
    """Make uncertainty propagation plot."""
    color = {
        "catboost_quadratic": "tab:blue",
        "quadratic": "tab:green",
        "quadratic_full": "tab:green",
        "neural_net": "tab:blue",
    }
    figures = {}
    a4_dims = (11.7, 8.27)
    for model in predictions:
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.kdeplot(benchmark, kernel="epa", lw=1.5, bw=0.01, ax=ax, color="black")
        sns.kdeplot(
            predictions[model],
            kernel="epa",
            lw=2.5,
            bw=0.01,
            alpha=0.8,
            ax=ax,
            color=color[model],
        )
        sns.despine(fig)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel("density")
        _ = ax.set_xlabel(r"$\Delta$ schooling")
        _ = ax.set_xlim(_get_xlim_via_quantiles(benchmark))
        figures[model] = fig
    return figures


def get_brute_force_models(project_name):
    if project_name == "thesis":
        out = ["brute_force_10000", "brute_force_70000"]
    else:
        out = ["brute_force_5000", "brute_force_15000"]
    return out


def uncertainty_propagation_plots_brute_force(benchmark, project_name):
    """Make uncertainty propagation plot."""
    color = {
        "brute_force_5000": "tab:orange",
        "brute_force_10000": "tab:orange",
        "brute_force_15000": "tab:red",
        "brute_force_70000": "tab:red",
    }
    figures = {}
    a4_dims = (11.7, 8.27)
    for model in get_brute_force_models(project_name):
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.kdeplot(benchmark, kernel="epa", lw=1.5, bw=0.01, ax=ax, color="black")
        brute_force = benchmark[: int(model.split("brute_force_")[1])].rename(
            "brute_force"
        )
        sns.kdeplot(
            brute_force,
            kernel="epa",
            lw=2.5,
            bw=0.01,
            alpha=0.8,
            ax=ax,
            color=color[model],
        )
        sns.despine(fig)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel("density")
        _ = ax.set_xlabel(r"$\Delta$ schooling")
        _ = ax.set_xlim(_get_xlim_via_quantiles(benchmark))
        figures[model] = fig
    return figures


def get_models(project_name):
    if project_name == "thesis":
        models = {
            "kw_94_one-neuralnet_large_thesis-70000": "neural_net",
            "kw_94_one-quadratic_full-10000": "quadratic",
        }
    elif project_name == "paper_uncertainty-propagation":
        models = {
            "catboost_quadratic_paper-15000": "catboost_quadratic",
            "quadratic_n-5000": "quadratic",
        }
    return models


def create_predictions(samples, project_name):
    models = get_models(project_name)
    fitted_models = {}
    for model, name in models.items():
        fitted_models[name] = surrogate.load(BLD / "surrogates" / project_name / model)

    if project_name == "thesis":
        col = "qoi500"
    else:
        col = "qoi"
    X = samples.drop(col, axis=1)
    predictions = {}
    for name, fitted_model in fitted_models.items():
        predictions[name] = surrogate.predict(X, fitted_model).flatten()

    predictions = pd.DataFrame.from_dict(predictions)
    return predictions


def save_figures(figures, project_name):
    """Save figure to bld path."""
    path = BLD / "figures" / project_name
    path.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(path / name, bbox_inches="tight")


def _get_xlim_via_quantiles(values, left=0.001, right=0.999):
    """Get floored and ceiled quantiles."""
    left, right = values.quantile([left, right])
    left = np.floor(left * 10) / 10
    right = np.ceil(right * 10) / 10
    return left, right


def load_samples(project_name):
    specifications = read_specifications()
    data_set = specifications[project_name]["data_set"][0]
    samples = pd.read_pickle(SRC / "data" / f"samples-{data_set}.pkl")
    return samples


def main(project_name):
    samples = load_samples(project_name)
    predictions = create_predictions(samples, project_name)
    col = "qoi" if project_name != "thesis" else "qoi500"
    benchmark = samples[col].rename("benchmark")
    surrogate_figures = uncertainty_propagation_plots_surrogates(benchmark, predictions)
    brute_force_figures = uncertainty_propagation_plots_brute_force(
        benchmark, project_name
    )
    figures = {**surrogate_figures, **brute_force_figures}
    save_figures(figures, project_name)


if __name__ == "__main__":
    keys = list(read_specifications().keys())
    for project_name in keys:
        main(project_name)
