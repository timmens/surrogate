================
Surrogate Models
================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


-------------------------
What is this project for?
-------------------------

The above code can be used to fit many statistical models to many data sets. In the
current set up each data set is split in training and testing. All specified models are
then fit to the training set and evaluated on the testing set. So overall the code
produces predictions of many statistical models on many testing sets. These predictions
can then be used for further analysis.


-----------------------
How to use this project
-----------------------

The general usage of the project is as follows. First you clone this repository to your
local machine. I then expect that you have `conda <https://docs.conda.io/en/latest/>`_
installed. At last you open up your favorite terminal emulator and run (line by line):

.. code-block:: zsh

    $ conda env create -f environment.yml
    $ conda activate surrogate
    $ pytask build


`pytask <https://pytask-dev.readthedocs.io/en/latest/index.html>`_ will then build the
project automatically. Once pytask is done all results are stored in the folder ``bld``.
The results are comprised of splitted data sets, fitted surrogate models and
predictions of fitted surrogate models on the testing data set; all of which can be
reused in further analysis.


Specify "projects"
==================

In the folder ``src/specs`` you find yaml specification files that can be used to
specify a problem. A typical file looks as follows:

.. code-block:: yaml

    run: True
    data_set:
        - kw_97_extended
    surrogate_type:
        - linear
        - quadratic
    n_obs:
        - 100
        - 200
        - 300
        - 400
        - 500


When this specification is read the build system will fit a "linear" and "quadratic"
model to the "kw_97_extended" data set using 100 to 500 training observations for each
model. If run is set to False the specification will be ignored. *Any new specification
file needs to have the exact same syntax!* Note also that this specification expects
a data set named ``samples-kw_97_extended.pkl`` in the folder ``src/data`` (see below).
Further, the specification expects the models "linear" and "quadratic" to be specified
in the file ``name_to_kwargs.yaml`` in the folder ``src/surrogates`` (see below).
At last note that one can create arbitrary many specification files; if the program is
build all specifications are read and executed iteratively.


Adding a new data set
=====================

In case you want to use the project with a new data set you have to move the new data
set in the folder ``src/data`` and name it ``samples-{identifier_of_data_set}.pkl`` (
note that the data has to be in a pickle format). In the current implementation the
code expects the outcome column to be named "qoi" and **all** other columns to be
features. *(Generalizing this implementation so that the outcome and feature columns
can be flexibly specified can be done if necessary; please submit a feature request or
create a pull request directly, if you need this feature.)* Note that this means that
the program will break if there are multiple columns that start with "qoi".

Since the training-testing split is done automatically one has to specify the number of
testing examples. For a new data set this has to be added to the dictionary in the file
``task_train_test_split.py``.


Creating a new surrogate model
==============================

Surrogate models
----------------

Surrogate models are specified in the file ``name_to_kwargs.yaml`` in the folder
``src/surrogates``. A typical entry looks as follows:

.. code-block:: yaml

    quadratic:
        model: PolynomialRegressor
        kwargs:
            degree: 2
            fit_intercept: True
            interaction: False
            scale: True
            degree: 2


The unique identifier of this model is "catboost_quadratic". The model then specifies
the surrogate module (see below) which is used for fitting and the kwargs specify the
keyword arguments that are used for the fitting procedure. For example if degree would
be 1 instead of 2, internally we would fit a model using only first order terms instead
of first order **and** second order terms. To add a new model you simply add such a
text-block to the end of the file ``name_to_kwargs.yaml``.


Surrogate modules
-----------------

The base models that can be used in conjunction with their keyword arguments to specify
a surrogate model (see above) are implemented in modules in the folder
``src/surrogates``. These modules have to export a ``fit`` and a ``predict`` function.


Adding a new module
-------------------

To add a new module you simply write up a new module with the corresponding functions
and integrate it into the module ``generic.py``. It can then be used in the
specifications files.
