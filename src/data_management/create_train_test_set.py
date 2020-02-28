import pickle

from sklearn.model_selection import train_test_split

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    # load original data
    df = pickle.load(open(ppj("IN_DATA", "data.pkl"), "rb"))

    # create and save training and testing data set
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=1)

    pickle.dump(df_train, open(ppj("OUT_DATA", "data_train.pkl"), "wb"))
    pickle.dump(df_test, open(ppj("OUT_DATA", "data_test.pkl"), "wb"))
