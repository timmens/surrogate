"""Provides custom class to transform features before fitting procedure.

The class can be used seamlessly with all sklearn methods but also as a standalone.


Usage (Example on how to scale features as well as to append 2nd order terms):

```
    X = get_data()
    ft = FeatureTransformer(degree=2, interaction=False, scale=True)
    X_transformed = ft.fit_transform(X)
```

"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom PolynomialFeatures which can exclude interaction terms."""

    def __init__(self, degree=1, interaction=True, scale=True):
        self.degree = degree
        self.interaction = interaction
        self.scale = scale

    def fit(self, X, y=None):
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False).fit(X)
        XX = _polynomial_features(X, self.degree, self.interaction, self.poly)
        self.scaler = StandardScaler().fit(XX)
        return self

    def transform(self, X, y=None):
        XX = _polynomial_features(X, self.degree, self.interaction, self.poly)
        if self.scale:
            XX = self.scaler.transform(XX)
        return XX

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def _polynomial_features(X, degree, interaction, poly):
    if interaction:
        XX = poly.transform(X)
    else:
        XX = np.concatenate([X ** k for k in range(1, degree + 1)], axis=1)
    return XX
