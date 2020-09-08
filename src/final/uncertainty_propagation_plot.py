import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.surrogates as surrogate
from src.config import BLD
from src.config import SRC


def uncertainty_propagation_plots_surrogates(benchmark, predictions):
    """Make uncertainty propagation plot."""
    color = {"catboost_quadratic": "tab:blue", "quadratic": "tab:green"}
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


def uncertainty_propagation_plots_brute_force(benchmark):
    """Make uncertainty propagation plot."""
    color = {"brute_force_5000": "tab:orange", "brute_force_15000": "tab:red"}
    figures = {}
    a4_dims = (11.7, 8.27)
    for model in ["brute_force_5000", "brute_force_15000"]:
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


def create_predictions(samples):
    models = {
        "catboost_quadratic_paper_n-15000": "catboost_quadratic",
        "quadratic_n-5000": "quadratic",
    }
    fitted_models = {}
    for model, name in models.items():
        fitted_models[name] = surrogate.load(
            BLD / "surrogates" / "paper_uncertainty-propagation" / model
        )

    X = samples.drop("qoi", axis=1)
    predictions = {}
    for name, fitted_model in fitted_models.items():
        predictions[name] = surrogate.predict(X, fitted_model).flatten()

    predictions = pd.DataFrame.from_dict(predictions)
    return predictions


def save_figures(figures):
    """Save figure to bld path."""
    path = BLD / "figures" / "paper_uncertainty-propagation"
    path.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(path / name, bbox_inches="tight")


def _get_xlim_via_quantiles(values, left=0.001, right=0.999):
    """Get floored and ceiled quantiles."""
    left, right = values.quantile([left, right])
    left = np.floor(left * 10) / 10
    right = np.ceil(right * 10) / 10
    return left, right


def load_samples():
    samples = pd.read_pickle(SRC / "data" / "samples-kw_97_extended.pkl")
    return samples


def main():
    samples = load_samples()
    predictions = create_predictions(samples)
    benchmark = samples["qoi"].rename("benchmark")
    surrogate_figures = uncertainty_propagation_plots_surrogates(benchmark, predictions)
    brute_force_figures = uncertainty_propagation_plots_brute_force(benchmark)
    figures = {**surrogate_figures, **brute_force_figures}
    save_figures(figures)


if __name__ == "__main__":
    main()
