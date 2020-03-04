"""Load implemented models and fit to training data set.

TODO: line 30, file endings dynamic.
"""
from bld.project_paths import project_paths_join as ppj
from src.analysis.auxiliary import get_surrogate_instances
from src.auxiliary.auxiliary import get_model_class_names
from src.auxiliary.auxiliary import load_implemented_models
from src.auxiliary.auxiliary import load_surrogate_fit_params
from src.data_management.auxiliary import load_training_data


if __name__ == "__main__":

    surrogates = load_implemented_models()
    surrogate_class_names = get_model_class_names(surrogates)
    surrogate_classes = get_surrogate_instances(surrogates)
    X, y = load_training_data()
    surrogate_params = load_surrogate_fit_params()

    for i, surrogate in enumerate(surrogate_classes):
        surrogate.fit(X, y, **surrogate_params[surrogate_class_names[i]])
        surrogate.save(ppj("OUT_ANALYSIS", surrogates[i] + ".csv"), overwrite=True)
