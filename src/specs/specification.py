"""Model specification."""
from collections import namedtuple

Specification = namedtuple(
    "Specification", ["model", "identifier", "n_obs", "fit_kwargs", "predict_kwargs"]
)
