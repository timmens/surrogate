"""Produces latex table which compares empirical quantiles of surrogates."""
import pandas as pd

import src.surrogates as surrogate
from src.config import BLD
from src.config import SRC
from src.specs import read_specifications


def save_quantiles_latex_table(quantiles_dict, project_name):
    df = pd.DataFrame.from_dict(quantiles_dict)
    df = df.T.round(decimals=4)
    path = BLD / "latex"
    path.mkdir(parents=True, exist_ok=True)
    df.to_latex(path / f"quantiles-{project_name}.tex")


def compute_quantiles(benchmark, predictions, quantiles=None):

    # quantiles = np.linspace(0, 1, num=51)
    quantiles = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
    results = {}
    results["quantiles"] = quantiles
    results["benchmark"] = benchmark.quantile(quantiles)
    results["brute-force5K"] = benchmark[:5000].quantile(quantiles)
    results["brute-force15K"] = benchmark[:15000].quantile(quantiles)
    for col in predictions:
        results[col] = predictions[col].quantile(quantiles)
    return results


def get_models(project_name, data_set):
    if project_name == "thesis":
        models = {
            "kw_94_one-neuralnet_large_thesis-70000": "neural_net",
            "kw_94_one-quadratic_full-10000": "quadratic",
        }
    elif project_name == "paper_uncertainty-propagation":
        models = {
            f"{data_set}-catboost_quadratic_paper-15000": "catboost_quadratic",
            f"{data_set}-quadratic-5000": "quadratic",
        }
    return models


def create_predictions(samples, project_name, data_set):
    models = get_models(project_name, data_set)
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


def load_samples(project_name):
    specifications = read_specifications()
    data_set = specifications[project_name]["data_set"][0]
    samples = pd.read_pickle(SRC / "data" / f"samples-{data_set}.pkl")
    return samples, data_set


def main(project_name):
    samples, data_set = load_samples(project_name)
    predictions = create_predictions(samples, project_name, data_set)
    col = "qoi" if project_name != "thesis" else "qoi500"
    benchmark = samples[col].rename("benchmark")
    quantiles_dict = compute_quantiles(benchmark, predictions)
    save_quantiles_latex_table(quantiles_dict, project_name)


if __name__ == "__main__":
    keys = list(read_specifications().keys())
    for project_name in keys:
        main(project_name)
