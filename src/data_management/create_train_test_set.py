import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    # load original data
    df = pickle.load(open(ppj("IN_DATA", "data.pkl"), "rb"))

    # rename outcomes (qoi) and features
    old_columns = df.columns

    is_outcome = old_columns.str.startswith("qoi")
    new_outcomes = ["y" + str(i + 1) for i in range(is_outcome.sum())]
    new_features = ["x" + str(i) for i in range((~is_outcome).sum())]

    new_columns = pd.Index(new_features + new_outcomes)
    df.columns = new_columns

    # save variable name reference
    variable_reference = pd.DataFrame(
        zip(old_columns, new_columns), columns=["old_name", "new_name"]
    )
    variable_reference.to_csv(ppj("OUT_DATA", "variable_reference.csv"), index=False)

    # create and save training and testing data set
    df_train, df_test = train_test_split(df, test_size=0.25)

    pickle.dump(df_train, open(ppj("OUT_DATA", "data_train.pkl"), "wb"))
    pickle.dump(df_test, open(ppj("OUT_DATA", "data_test.pkl"), "wb"))
