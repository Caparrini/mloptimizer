Parallel processing
===================

``mloptimizer`` supports parallel evaluation of individuals using `joblib <https://joblib.readthedocs.io/>`_.
This can significantly speed up optimization, especially for models with longer training times.

Enabling Parallel Processing
----------------------------

Use the ``use_parallel`` parameter when initializing :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>`:

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from sklearn.tree import DecisionTreeClassifier

    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Enable parallel processing (default is True)
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        use_parallel=True  # Default
    )

    # Disable parallel processing for sequential execution
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        use_parallel=False
    )

Performance Comparison Example
------------------------------

Here's an example comparing parallel vs. sequential execution:

.. code-block:: python

    import time
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # With parallel processing
    opt_parallel = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=5,
        population_size=20,
        use_parallel=True,
        seed=42
    )

    start = time.time()
    opt_parallel.fit(X, y)
    time_parallel = time.time() - start

    # Without parallel processing
    opt_sequential = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=5,
        population_size=20,
        use_parallel=False,
        seed=42
    )

    start = time.time()
    opt_sequential.fit(X, y)
    time_sequential = time.time() - start

    print(f"Parallel: {time_parallel:.2f}s")
    print(f"Sequential: {time_sequential:.2f}s")

.. note::

    The speedup from parallel processing depends on:

    - **Model training time**: Larger speedups for models that take longer to train
    - **Number of CPU cores**: More cores = more potential parallelism
    - **Population size**: Larger populations benefit more from parallelization
    - **Overhead**: For very fast models (like DecisionTree on small data), the parallelization overhead may outweigh benefits

MLflow Compatibility
--------------------

MLflow tracking works seamlessly with parallel execution. When ``use_parallel=True``, mloptimizer:

1. Collects evaluation metadata from parallel workers
2. Logs child runs in batches after each generation completes
3. Creates the same comprehensive MLflow data as sequential mode

.. code-block:: python

    # Full MLflow logging works with parallel execution
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        use_mlflow=True,
        use_parallel=True  # Works with MLflow (default)
    )

.. note::

    Child runs are logged after each generation completes rather than in real-time.
    This is transparent to the user - all data appears correctly in MLflow UI.

When to Disable Parallel Processing
-----------------------------------

Consider setting ``use_parallel=False`` when:

- Debugging optimization issues (easier to trace sequential execution)
- Running on systems with limited memory (parallel workers increase memory usage)
- The model itself uses parallelization (e.g., ``n_jobs=-1`` in RandomForest)
