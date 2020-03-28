"""Load specific model and fit to training data set."""
import sys

import numpy as np

from bld.project_paths import project_paths_join as ppj
from src.auxiliary.auxiliary import get_surrogate_instances
from src.auxiliary.auxiliary import load_sorted_features
from src.auxiliary.auxiliary import load_surrogates_specs
from src.data_management.utilities import load_training_data


if __name__ == "__main__":
    model_name = sys.argv[1]

    # load all models specifications
    specs = load_surrogates_specs()
    # extract specific model
    model_specs = specs[model_name]

    # extract model class name
    name = model_specs["model"]

    # extract model fit kwargs
    kwargs = model_specs["kwargs"]

    # initiate model classes
    model = get_surrogate_instances(name)

    # load order of importance of features
    ordered_features = load_sorted_features()

    # subset features
    nfeatures = model_specs["nfeatures"]
    if isinstance(nfeatures, int):
        feature_list = ordered_features[:nfeatures]
    else:
        max_num_features = len(ordered_features)
        num_features = int(nfeatures[len("random") :])
        np.random.seed(seed=1)
        feature_index = np.random.choice(
            range(num_features), size=num_features, replace=False
        )
        feature_list = ordered_features[feature_index]

    # load training data
    nobs = model_specs["nobs"]
    X, y = load_training_data(nobs=nobs, seed=1)

    model.fit(X[feature_list], y, **kwargs)
    model.save(ppj("OUT_ANALYSIS", model_name), overwrite=True)
