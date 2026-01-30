Early Stopping
==============

Early stopping automatically terminates the optimization when no significant
improvement is observed, saving computation time.

Enabling Early Stopping
-----------------------

Use the ``early_stopping``, ``patience``, and ``min_delta`` parameters:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=50,  # Maximum generations
        population_size=10,
        early_stopping=True,
        patience=5,      # Stop after 5 generations without improvement
        min_delta=0.001  # Minimum improvement threshold
    )
    opt.fit(X, y)

Parameters
----------

``early_stopping`` : bool, default=False
    Whether to enable early stopping.

``patience`` : int, default=5
    Number of generations without improvement before stopping.
    The optimization stops if no improvement greater than ``min_delta`` is
    observed for this many consecutive generations.

``min_delta`` : float, default=0.001
    Minimum improvement in fitness required to reset the patience counter.
    Set higher values for noisy fitness functions.

How It Works
------------

1. After each generation, the best fitness is compared to the previous best
2. If improvement is greater than ``min_delta``, the patience counter resets
3. If improvement is less than ``min_delta``, the patience counter increments
4. When the counter reaches ``patience``, optimization stops early

Example: Comparing With and Without
-----------------------------------

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Without early stopping - runs all generations
    opt_full = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=20,
        population_size=10,
        early_stopping=False,
        seed=42
    )
    opt_full.fit(X, y)

    # With early stopping - may stop earlier
    opt_early = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=20,
        population_size=10,
        early_stopping=True,
        patience=3,
        seed=42
    )
    opt_early.fit(X, y)

    print(f"Without early stopping: {opt_full.n_trials_} evaluations")
    print(f"With early stopping: {opt_early.n_trials_} evaluations")

When to Use Early Stopping
--------------------------

**Enable early stopping when:**

- Running long optimizations where convergence is expected
- Training expensive models (RandomForest, XGBoost, neural networks)
- Searching large hyperparameter spaces
- Running batch experiments with many optimizations

**Disable early stopping when:**

- You need deterministic number of evaluations
- Exploring highly multimodal fitness landscapes
- The fitness function is very noisy

Tuning Tips
-----------

- **Start with patience=5**: Good default for most problems
- **Increase patience for noisy metrics**: Use patience=10-15 if fitness varies
- **Adjust min_delta for scale**: For accuracy (0-1), use 0.001-0.01; for MSE, scale accordingly
- **Use with verbose=1**: Monitor when early stopping triggers

.. seealso::

    :doc:`/auto_examples/plot_early_stopping` for a complete example.
