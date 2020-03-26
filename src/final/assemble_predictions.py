import pickle

import pandas as pd

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":

    # assemble final data frame
    df_test = pickle.load(open(ppj("OUT_DATA", "data_test.pkl"), "rb"))
    df_test = df_test.reset_index()

    to_drop = ["qoi_tuition_subsidy_1000", "qoi_tuition_subsidy_1500"]
    df_test = df_test.drop(to_drop, axis=1)

    predictions = pickle.load(open(ppj("OUT_DATA", "predictions.pkl"), "rb"))

    df_out = pd.concat((df_test, predictions), axis=1, sort=False)

    # save data frame
    pickle.dump(df_out, open(ppj("OUT_DATA", "data_with_predictions.pkl"), "wb"))
