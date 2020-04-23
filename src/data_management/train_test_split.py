import pickle

import pandas as pd

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    # load original data
    df = pickle.load(open(ppj("IN_DATA", "data.pkl"), "rb"))

    to_drop = ["qoi_tuition_subsidy_1000", "qoi_tuition_subsidy_1500"]
    df = df.drop(to_drop, axis=1)

    # create and save training and testing data set
    df_validation = df.iloc[:25000]
    df_validation.index = pd.MultiIndex.from_product(
        [["validation"], range(len(df_validation))], names=["dataset", "iteration"]
    )

    df_training = df.iloc[25000:]
    df_training.index = pd.MultiIndex.from_product(
        [["training"], range(len(df_training))], names=["dataset", "iteration"]
    )

    pickle.dump(df_validation, open(ppj("OUT_DATA", "df_validation.pkl"), "wb"))
    pickle.dump(df_training, open(ppj("OUT_DATA", "df_training.pkl"), "wb"))
