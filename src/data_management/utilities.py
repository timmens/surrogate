"""Helper functions for data loading and manipulation.

Here we assume that the original data frame has its outcomes labelled by having
"qoi_" at the beginning of the column name.
"""
import pickle

import numpy as np

from bld.project_paths import project_paths_join as ppj


def extract_features_from_data(df):
    """Extract features from data frame.

    Args:
        df (pd.DataFrame): A data frame with columns as in the original df.

    Returns:
        X (pd.DataFrame): A data frame containing only features.

    """
    feature_column = ~df.columns.str.startswith("qoi")
    X = df.loc[:, feature_column].copy()
    return X


def extract_outcome_from_data(df, outcome=1):
    """Extract specific utcome from data frame.

    Args:
        df (pd.DataFrame): A data frame with columns as in the original df.
        outcome (int): Which outcome to select. Possible values: 1, 2, 3.

    Returns:
        y (np.ndarray): The specified outcome.

    """
    outcome_column = "qoi_tuition_subsidy_" + str(500 * outcome)
    return df[outcome_column].values


def load_testing_data(nobs=25000):
    """Load data for testing.

    Args:
        nobs (int): The number of testing observations.
            Has to be in the range [0, 25000].

    Returns:
        X, y (np.array): Testing features and testing outcomes.

    """
    df = pickle.load(open(ppj("OUT_DATA", "data_test.pkl"), "rb"))

    df = df.iloc[:nobs, :]

    Xtest = extract_features_from_data(df)
    ytest = extract_outcome_from_data(df)

    return Xtest, ytest


def load_training_data(nobs=75000, seed=1):
    """Load data for testing and training.

    Args:
        nobs (int): The number of training observations.
            Has to be in the range [0, 75000].
        seed (int): Random number seed.

    Returns:
        X, y (np.array): Training features and Training outcomes.
    """
    df = pickle.load(open(ppj("OUT_DATA", "data_train.pkl"), "rb"))

    np.random.seed(seed)
    index = np.random.choice(np.arange(75000), size=nobs, replace=False)

    df = df.iloc[index, :]

    X = extract_features_from_data(df)
    y = extract_outcome_from_data(df)

    return X, y


def get_feature_names(poly, X):
    """Extract feature names of polynomial features.

    Args:
        poly: fitted ``sklearn.preprocessing.PolynomialFeatures`` object.
        X (pd.DataFrame): Data on features.

    Returns:
        coef_names (list): List of feature names corresponding to all polynomials.
    """
    coef_names = poly.get_feature_names(X.columns)
    coef_names = [name.replace(" ", ":") for name in coef_names]
    return coef_names


def compute_testing_loss(model, ytest, Xtest, measure, **kwargs):
    ypred = model.predict(Xtest, **kwargs)
    mae = measure(ytest, ypred)
    return mae
