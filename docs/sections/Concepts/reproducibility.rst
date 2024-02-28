Reproducibility
===================

Reproducibility is a key aspect of
scientific research, and more precisely,
in machine learning. ``MLOptimizer`` provides an
input parameter ``seed`` that allows to set
the random seed for:

- The random number generator of the optimizer generating the initial population and the mutations
- The random number generator of the model on training
- The random number generator of the data on split

An example of usage is:

.. code-block:: python

    from sklearn.datasets import load_breast_cancer as dataset
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from mloptimizer.genoptimizer import TreeOptimizer

    X, y = load_iris(return_X_y=True)
    population = 2
    generations = 2
    seed = 25
    distinct_seed = 2
    # It is important to run the optimization
    # right after the creation of the optimizer
    optimizer1 = TreeOptimizer(X, y, seed=seed)
    result1 = optimizer1.optimize_clf(population=population,
                                      generations=generations)
    # WARNING: In case the optimizer2 would be created after the optimizer1,
    # the results would be different
    optimizer2 = TreeOptimizer(X, y, score_function=target_metric,
                           eval_function=target_score, seed=seed)
    result2 = optimizer2.optimize_clf(population=population,
                                      generations=generations)

    optimizer3 = TreeOptimizer(X, y, score_function=target_metric,
                           eval_function=target_score, seed=distinct_seed)
    result3 = optimizer3.optimize_clf(population=population,
                                      generations=generations)
    str(result1) == str(result2)
    str(result1) != str(result3)

.. warning::

    To ensure reproducibility, it is important to run the optimization
    right after the creation of the optimizer with the seed to ensure no
    other random number generator has been used in the meantime.