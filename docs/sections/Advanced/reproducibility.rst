Reproducibility
===============

Reproducibility is essential for scientific research and machine learning experiments.
``mloptimizer`` provides a ``seed`` parameter to ensure consistent results across runs.

Using the Seed Parameter
------------------------

Set the ``seed`` parameter when initializing :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>`:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Same seed = same results
    opt1 = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=5,
        population_size=10,
        seed=42
    )
    opt1.fit(X, y)

    opt2 = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=5,
        population_size=10,
        seed=42  # Same seed
    )
    opt2.fit(X, y)

    # Results should be identical
    assert opt1.best_params_ == opt2.best_params_
    assert opt1.best_score_ == opt2.best_score_

What the Seed Controls
----------------------

The ``seed`` parameter sets the random state for:

- **Initial population generation**: The random hyperparameter values for the first generation
- **Genetic operations**: Selection, crossover, and mutation operations
- **Model training**: Passed to estimators that support ``random_state`` parameter
- **Cross-validation splits**: When using CV-based evaluation

Example: Verifying Reproducibility
----------------------------------

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    results = []
    for _ in range(3):
        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=3,
            population_size=8,
            seed=123  # Fixed seed
        )
        opt.fit(X, y)
        results.append(opt.best_score_)

    # All results should be identical
    assert all(r == results[0] for r in results)
    print(f"All runs produced the same result: {results[0]}")

Different Seeds for Different Results
-------------------------------------

Use different seeds to explore different optimization paths:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    best_scores = []
    for seed in [1, 2, 3, 4, 5]:
        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=5,
            population_size=10,
            seed=seed
        )
        opt.fit(X, y)
        best_scores.append(opt.best_score_)
        print(f"Seed {seed}: best_score = {opt.best_score_:.4f}")

    print(f"\nBest overall: {max(best_scores):.4f}")

.. note::

    Running multiple optimizations with different seeds and taking the best result
    is a common strategy to improve the chances of finding a good solution.

Limitations
-----------

.. warning::

    Reproducibility may be affected by:

    - **Parallel processing**: When ``use_parallel=True``, the order of parallel evaluations may vary slightly between runs on different hardware
    - **External randomness**: Some estimators may have internal randomness not controlled by the seed
    - **Floating point precision**: Minor differences in floating point arithmetic across platforms

    For guaranteed reproducibility, consider using ``use_parallel=False``.
