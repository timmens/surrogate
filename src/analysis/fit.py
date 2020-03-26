"""Load specified models and fit to training data set."""
from bld.project_paths import project_paths_join as ppj
from src.auxiliary.auxiliary import get_surrogate_instances
from src.auxiliary.auxiliary import load_sorted_features
from src.auxiliary.auxiliary import load_surrogates_specs
from src.data_management.utilities import load_training_data


if __name__ == "__main__":

    # load dictionary of model specifications
    specs = load_surrogates_specs()

    # extract model names
    names = [specs[key]["model"] for key in specs.keys()]

    # extract model fit kwargs
    params = [specs[key]["kwargs"] for key in specs.keys()]

    # initiate model classes
    classes = get_surrogate_instances(names)

    # load training data
    X, y = load_training_data(seed=1)

    # load order of importance of features
    ordered_features = load_sorted_features()

    for key, kwargs, model in zip(list(specs.keys()), params, classes):
        features = specs[key]["features"]
        features = features if features != 0 else len(ordered_features)
        feature_list = ordered_features[:features]

        model.fit(X[feature_list], y, **kwargs)
        model.save(ppj("OUT_ANALYSIS", key), overwrite=True)
