===========================
Metrics and Score Functions
===========================

This section explains how to customize the scoring metric used during genetic optimization.

Default Scoring
---------------

By default, :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` uses:

- **Classification**: ``accuracy`` (from sklearn)
- **Regression**: ``neg_mean_squared_error`` (from sklearn)

Using Built-in Scoring Metrics
------------------------------

Use the ``scoring`` parameter to specify any sklearn-compatible scoring metric:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Use balanced accuracy instead of regular accuracy
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        scoring='balanced_accuracy',
        generations=5,
        population_size=10
    )
    opt.fit(X, y)

Common Scoring Metrics
----------------------

**Classification metrics:**

- ``accuracy`` - Accuracy score
- ``balanced_accuracy`` - Balanced accuracy (good for imbalanced datasets)
- ``f1`` - F1 score (binary)
- ``f1_weighted`` - Weighted F1 score (multiclass)
- ``precision`` - Precision score
- ``recall`` - Recall score
- ``roc_auc`` - ROC AUC score

**Regression metrics:**

- ``neg_mean_squared_error`` - Negative MSE (sklearn convention: higher is better)
- ``neg_root_mean_squared_error`` - Negative RMSE
- ``neg_mean_absolute_error`` - Negative MAE
- ``r2`` - R-squared score

.. note::

    sklearn uses the convention that higher scores are better. For error metrics like MSE,
    the negative value is used so that minimizing error = maximizing the negative error.

Custom Scoring Functions
------------------------

You can define custom scoring functions using sklearn's ``make_scorer``:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import make_scorer, fbeta_score
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Create a custom F-beta scorer with beta=2 (recall-weighted)
    f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        scoring=f2_scorer,
        generations=5,
        population_size=10
    )
    opt.fit(X, y)

Example: Custom Business Metric
-------------------------------

For domain-specific metrics, create a custom scoring function:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import make_scorer
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    # Custom metric: penalize false negatives more than false positives
    def weighted_error(y_true, y_pred):
        fn_weight = 2.0  # False negatives cost twice as much
        fp_weight = 1.0

        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        # Return a score where higher is better
        if tp + fn + fp == 0:
            return 1.0
        return tp / (tp + fn_weight * fn + fp_weight * fp)

    X, y = load_iris(return_X_y=True)
    # Binary classification for this example
    y_binary = (y == 2).astype(int)

    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)
    custom_scorer = make_scorer(weighted_error)

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        scoring=custom_scorer,
        generations=5,
        population_size=10
    )
    opt.fit(X, y_binary)

Cross-Validation Integration
----------------------------

The ``scoring`` parameter works with cross-validation. Use the ``cv`` parameter to specify the CV strategy:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Use 5-fold stratified CV with F1 scoring
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        generations=5,
        population_size=10
    )
    opt.fit(X, y)

    print(f"Best F1 score: {opt.best_score_:.4f}")

.. seealso::

    - `sklearn scoring documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_
    - `sklearn make_scorer <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`_
