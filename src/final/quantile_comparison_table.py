"""Produces latex table which compares empirical quantiles of surrogates."""
import pandas as pd

import src.surrogates as surrogate
from src.config import BLD
from src.config import SRC


def save_quantiles_latex_table(quantiles_dict):
    df = pd.DataFrame.from_dict(quantiles_dict)
    df = df.T.round(decimals=4)
    path = BLD / "latex"
    path.mkdir(parents=True, exist_ok=True)
    df.to_latex(path / "quantiles.tex")


def compute_quantiles(
    benchmark, predictions, quantiles=(0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
):
    results = {}
    results["quantiles"] = quantiles
    results["benchmark"] = benchmark.quantile(quantiles)
    results["brute-force5K"] = benchmark[:5000].quantile(quantiles)
    results["brute-force15K"] = benchmark[:15000].quantile(quantiles)
    for col in predictions:
        results[col] = predictions[col].quantile(quantiles)
    return results


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


def load_samples():
    samples = pd.read_pickle(SRC / "data" / "samples-kw_97_extended.pkl")
    return samples


def main():
    samples = load_samples()
    predictions = create_predictions(samples)
    benchmark = samples["qoi"].rename("benchmark")
    quantiles_dict = compute_quantiles(benchmark, predictions)
    save_quantiles_latex_table(quantiles_dict)


if __name__ == "__main__":
    main()
