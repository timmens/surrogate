import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.utilities.utilities import load_model_information


def main():
    losses = pd.read_csv(ppj("OUT_ANALYSIS", "losses.csv"))
    losses = losses.query("measure=='mae'")

    models = load_model_information()

    df = pd.DataFrame(columns=["method", "features", "samples", "mae"])
    df = df.set_index(["method", "features", "samples"])

    for model in models:
        q = f"method=='{model.colname}'"
        mae = losses.query(q)["loss"].values[0]
        df.loc[model.name, model.p, model.n] = mae

    df.to_csv(ppj("OUT_FINAL", "losses_mae_tidy.csv"))


if __name__ == "__main__":
    main()
