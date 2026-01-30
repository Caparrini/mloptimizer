Population Seeding (initial_params)
===================================

Population seeding allows you to initialize the genetic algorithm with known good
hyperparameter configurations, giving the optimization a "warm start."

Basic Usage
-----------

Use the ``initial_params`` parameter to provide a list of hyperparameter dictionaries:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Known good configurations from previous experiments
    initial_configs = [
        {'max_depth': 5, 'min_samples_split': 10},
        {'max_depth': 3, 'min_samples_split': 5},
    ]

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        initial_params=initial_configs,
        generations=10,
        population_size=10
    )
    opt.fit(X, y)

Parameters
----------

``initial_params`` : list of dict or None, default=None
    List of hyperparameter configurations to seed the initial population.
    Each dict maps hyperparameter names to values. Unspecified hyperparameters
    are randomly sampled from the search space.

``include_default`` : bool, default=False
    Whether to include sklearn's default hyperparameters in the initial population.
    This ensures you have at least one reasonable baseline configuration.

Including sklearn Defaults
--------------------------

Set ``include_default=True`` to automatically add sklearn's default values:

.. code-block:: python

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        initial_params=[{'max_depth': 5}],
        include_default=True,  # Also include sklearn defaults
        generations=10,
        population_size=10
    )

Partial Hyperparameter Specification
------------------------------------

You don't need to specify all hyperparameters. Unspecified ones are randomly sampled:

.. code-block:: python

    # Only specify max_depth, other params are random
    initial_configs = [
        {'max_depth': 5},
        {'max_depth': 10},
    ]

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        initial_params=initial_configs,
        generations=10,
        population_size=10
    )

Use Cases
---------

**1. Warm-starting from GridSearch results:**

.. code-block:: python

    from sklearn.model_selection import GridSearchCV

    # Run a quick GridSearch first
    param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3)
    grid.fit(X, y)

    # Use top results to seed genetic optimization
    top_params = [grid.best_params_]

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        initial_params=top_params,
        generations=20,
        population_size=15
    )

**2. Continuing from a previous optimization:**

.. code-block:: python

    # First optimization
    opt1 = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=5,
        population_size=10,
        seed=42
    )
    opt1.fit(X, y)

    # Continue with the best result
    opt2 = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        initial_params=[opt1.best_params_],
        generations=10,
        population_size=10,
        seed=43
    )
    opt2.fit(X, y)

**3. Exploring around known good regions:**

.. code-block:: python

    # Create variations around a known good config
    base_config = {'max_depth': 5, 'min_samples_split': 10}
    variations = [
        {'max_depth': 4, 'min_samples_split': 8},
        {'max_depth': 5, 'min_samples_split': 10},  # base
        {'max_depth': 6, 'min_samples_split': 12},
    ]

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        initial_params=variations,
        generations=10,
        population_size=10
    )

Best Practices
--------------

1. **Keep seeds smaller than population**: Leave room for random individuals to maintain genetic diversity

2. **Provide diverse configurations**: Don't cluster all seeds in one region of the search space

3. **Use with early stopping**: Seeding + early stopping can quickly converge to good solutions

4. **Combine with include_default**: Ensures you have a reasonable baseline

.. seealso::

    :doc:`/auto_examples/plot_initial_params` for a complete example.
