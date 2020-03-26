import pickle

import pandas as pd

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    res = pickle.load(open(ppj("IN_DATA", "kw_indices.pkl"), "rb"))

    df = res["mu_corr"].copy().to_frame(name="mu_corr")
    names = ["_".join(tup) for tup in df.index]
    df["name"] = names
    df.index = pd.Index(range(len(df)))

    sorted_features = df.sort_values(by="mu_corr", ascending=False)["name"]
    sorted_features.to_csv(
        ppj("OUT_DATA", "sorted_features.csv"), index=False, header=False
    )
