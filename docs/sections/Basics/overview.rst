=========================
Overview
=========================

Introduction
------------
The main class objects are the `Optimizer` and the `HyperparameterSpace` classes.

The optimizer `Optimizer` is able to optimize any model that complies with the `sklearn` API.
The `HyperparameterSpace` class is used to define the hyperparameters that will be optimized, either
the fixed hyperparameters or the hyperparameters that will be optimized.

Usage
-----
To use the `Optimizer` class:

1. Define your features and labels.
2. Choose a model to optimize that complies with the `sklearn` API. (e.g. `XGBClassifier`).
2. Create an instance of `HyperparameterSpace` with the hyperparameters that you want to optimize.
3. Call the `optimize_clf()` method to start the optimization process.

.. note::
    There are default HyperparameterSpaces defined in the ``conf`` folder for the most common models.
    You can use the HyperparameterSpace.get_default_hyperparams(class) (class e.g. XGBClassifier).

There are several parameters than can be passed to the `Optimizer` constructor:

- `estimator_class`: The class of the model to optimize. It should comply with the `sklearn` API.
- `X`: The features of your dataset.
- `y`: The labels of your dataset.
- `folder`: The folder where the files and folder will be saved. Defaults to the current directory.
- `log_file`: The name of the log file. Defaults to `mloptimizer.log`.
- `hyperparam_space`: The hyperparameter space to use for the optimization process.
- `eval_function`: The function to use to evaluate the model. Defaults to `train_score`.
- `score_function`: The function to use to score the model. Defaults to `accuracy_score`.
- `seed`: The seed to use for reproducibility. Defaults to a random integer between 0 and 1000000.


Default Usage Example
---------------------

The simplest example of using the Optimizer is:

- Store your features and labels in `X` and `y` respectively.
- Use HyperparameterSpace.get_default_hyperparams(XGBClassifier) to get the default hyperparameters for the model you want to optimize.
- Create an instance of `Optimizer` with your classifier class, hyperparameter space, data and leave all other parameters to their default values.
- Call the `optimize_clf()` method to start the optimization process. You can pass the population size and the number of generations to the method.
- The result of the optimization process will be a object of type XGBClassifier with the best hyperparameters found.

.. code-block:: python

    from mloptimizer.core import Optimizer
    from mloptimizer.hyperparams import HyperparameterSpace
    from xgboost import XGBClassifier
    from sklearn.datasets import load_iris

    # 1) Load the dataset and get the features and target
    X, y = load_iris(return_X_y=True)

    # 2) Define the hyperparameter space (a default space is provided for some algorithms)
    hyperparameter_space = HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier)

    # 3) Create the optimizer and optimize the classifier
    opt = Optimizer(estimator_class=XGBClassifier, features=X, labels=y, hyperparam_space=hyperparameter_space)

    clf = opt.optimize_clf(10, 10)

This will create a folder (in the current location) with name `YYYYMMDD_nnnnnnnnnn_Optimizer`
(where `YYYYMMDD_nnnnnnnnnn` is the current timestamp) and a log file named `mloptimizer.log`.
To inspect the structure of the folder and what can you find in it, please refer to the `Folder Structure` section.

Custom HyperparameterSpace Example
----------------------------------

Among the parameters that can be passed to the `Optimizer` constructor,
the `hyperaram_space` of class `HyperparameterSpace` is really important
and should be aligned with the machine learning algorithm passed to the Optimizer: `fixed_hyperparams`
and `evolvable_hyperparams`.

The `evolvable_hyperparams` parameter is a dictionary of custom hyperparameters.
The key of each hyperparameter is the name of the hyperparameter, and the value is the `Hyperparam` object itself.
To understand how to use the `Hyperparam` object, please refer to the `Hyperparam` section inside Concepts.

The `fixed_hyperparams` parameter is a dictionary of fixed hyperparameters.
This is simply a dictionary where the key is the name of the hyperparameter, and the value is the value of the hyperparameter.
These hyperparameters will not be optimized, but will be used as fixed values during the optimization process.

An example of using custom hyperparameters is:

.. code-block:: python

    # Define your custom hyperparameters
    fixed_hyperparams = {
        'max_depth': 5
    }
    evolvable_hyperparams = {
        'colsample_bytree': Hyperparam("colsample_bytree", 3, 10, 'float', 10),
        'gamma': Hyperparam("gamma", 0, 20, 'int'),
        'learning_rate': Hyperparam("learning_rate", 1, 100, 'float', 1000),
        # 'max_depth': Hyperparam("max_depth", 3, 20, 'int'),
        'n_estimators': Hyperparam("n_estimators", 100, 500, 'int'),
        'subsample': Hyperparam("subsample", 700, 1000, 'float', 1000),
        'scale_pos_weight': Hyperparam("scale_pos_weight", 15, 40, 'float', 100)
    }


    custom_hyperparam_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)

    # Create an instance of XGBClassifierOptimizer with custom hyperparameters
    xgb_optimizer = Optimizer(estimator_class=XGBClassifier,features=X, labels=y,
                              hyperparam_space=custom_hyperparam_space)

    # Start the optimization process
    result = xgb_optimizer.optimize_clf(3, 3)





Both `evolvable_hyperparams` and `fixed_hyperparams` can be used together,
providing several different ways to customize the optimization process.

Reproducibility
---------------

Researchers often need to be able to reproduce their results. During the research process it could be
advisable to run several optimizations processes with different parameters or input data.
However, if the results of the optimization process are not reproducible, it will be difficult to compare
the results of the different optimization processes.
In order to make the results reproducible, the `Optimizer` have a `seed` parameter.
This parameter is used to set the seed of the random number generator used during the optimization process.
If you set the same seed, the results of the optimization process will be the same.

An example of two executions of the optimization process with the same seed that will produce the same result is:

.. code-block:: python

    from mloptimizer.core import Optimizer
    from mloptimizer.hyperparams import HyperparameterSpace
    from xgboost import XGBClassifier
    from sklearn.datasets import load_iris

    # 1) Load the dataset and get the features and target
    X, y = load_iris(return_X_y=True)

    # 2) Define the hyperparameter space (a default space is provided for some algorithms)
    hyperparameter_space = HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier)

    # 3) Create two instances of Optimizer with the same seed
    xgb_optimizer1 = Optimizer(estimator_class=XGBClassifier, features=X, labels=y,
                               hyperparam_space = hyperparameter_space, seed=42)
    result1 = xgb_optimizer1.optimize_clf(3, 3)

    xgb_optimizer2 = Optimizer(estimator_class=XGBClassifier, features=X, labels=y,
                               hyperparam_space = hyperparameter_space, seed=42)
    result2 = xgb_optimizer2.optimize_clf(3, 3)

    # Verify that the results are the same
    # The comparison is done using the string representation of the result objects
    # which are the hyperparameters of the best model found
    assert str(result1)== str(result2)

