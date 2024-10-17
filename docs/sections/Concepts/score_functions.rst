===========================
Metrics and score functions
===========================

.. note::
   This section provides an overview of metrics and score functions and how they are used in the context of genetic optimization. It also provides examples of how to use the score functions provided by the library.

Introduction
------------

Metrics and score functions are used to evaluate the performance of machine learning algorithms. They take as input the true labels\values of the data, the predicted labels\values of the data, a metric (e.g. accuracy, precision, rmse, mse, etc.)
and a scoring function or strategy (train/test split, k-fold cross validation, stratified cross validation, time series split, etc.). The score function then calculates a score that quantifies how well the machine learning algorithm performed on the given data.
Two main types of metrics are commonly used in machine learning:

- Classification metrics: These are used to evaluate the performance of classification algorithms. They include metrics such as accuracy, precision, recall, F1 score, etc.

- Regression metrics: These are used to evaluate the performance of regression algorithms. They include metrics such as mean squared error, mean absolute error, R-squared, etc.

In the context of genetic optimization, the fitness score is the metric aimed to be maximized. When passing a classification or regression estimator mloptimizer will use by default the following score functions:

- Classification: balanced_accuracy_score
- Regression: rmse

In the case of the regression metrics, we maximize the negative of the metric, so the optimization is done in the same way as in the classification case.

However, the user can pass any score function.

Metrics
-------

The `metrics` input argument in the `Optimizer` class is a dictionary
that maps a metric name to a metric function. This function can be one of the metrics provided
by the `sklearn.metrics` module, or a custom metric function that shoutd comply with the sklearn library metric functions.

Here's an example of how to use the `Optimizer` class with custom metrics:

.. code-block:: python

    from sklearn.metrics import balanced_accuracy_score, mean_squared_error
    from mloptimizer.application import Optimizer
    from mloptimizer.domain.hyperspace import HyperparameterSpace
    from sklearn.ensemble import RandomForestRegressor

    regression_metrics = {
            "mse": mean_squared_error,
            "rmse": root_mean_squared_error
        }
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(RandomForestRegressor)

    mlopt = Optimizer(estimator_class=RandomForestRegressor,
                      hyperparam_space=evolvable_hyperparams,
                      fitness_score='rmse', metrics=regression_metrics,
                      features=X, labels=y)

Score Functions
---------------

Not only the metric should be defined, how the data is used or split to calculate the metric is also important.
The `model_evaluation.py` module provides score functions that can be used to evaluate the performance of machine learning algorithms.
These score functions take a estimator, features, labels, and a score metric as input, and return a score that quantifies how well the classifier performed on the given data.

The `model_evaluation.py` module provides the following score functions:

- `train_score`: This function trains a classifier with the provided features and labels, and then calculates the score over the train data.

- `train_test_score`: This function splits the provided features and labels into a training set and a test set. Then, it trains an estimator on the training set and calculates a score on the test set using the provided score function.

- `kfold_score`: This function evaluates an estimator using K-Fold cross-validation. It splits the provided features and labels into K folds, trains an estimator on K-1 folds, and calculates a score on the remaining fold. This process is repeated K times, and the function returns the average score across all folds.

- `kfold_stratified_score`: This function is similar to `kfold_score`, but it uses stratified K-Fold cross-validation. This means that it preserves the percentage of samples for each class in each fold. For classification problems, this can help ensure that each fold has a representative sample of each class.

- `temporal_kfold_score`: This function is similar to `kfold_score`, but it uses temporal K-Fold cross-validation. This means that it respects the order of the data, making it suitable for time series data in order to avoid look-ahead bias.

Each of these score functions takes a classifier, features, and labels as input. They also take a score metric as input, which is used to calculate the score. The score function could be any function that takes the true labels\values and the predicted labels\values as input and returns a score. Examples of score functions include accuracy, precision, recall, F1 score, etc.

Examples
--------

Here's an example of how to use the `train_score` function:

.. code-block:: python

   from mloptimizer.domain.evaluation import model_evaluation
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score

   # Define features, labels, and classifier
   from sklearn.datasets import load_iris
   features, labels = load_iris(return_X_y=True)
   clf = RandomForestClassifier()

   # Use the train_score function
   score = model_evaluation.train_score(features, labels, clf, metrics={"accuracy": accuracy_score})


In this example, we first define the features, labels, and classifier. We then use the `train_score` function to train the classifier and calculate the score. The `accuracy_score` function from `sklearn.metrics` is used as the score function.

