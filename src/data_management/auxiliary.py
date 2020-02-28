"""Helper functions for data manipulation.

Here we assume that the original data frame has its outcomes labelled by having
"qoi_" at the beginning of the column name.
"""


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
