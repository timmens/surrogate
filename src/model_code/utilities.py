"""Utilities for model code."""


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
