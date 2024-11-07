Overview
=========================

`mloptimizer` provides a flexible framework for optimizing machine learning models through hyperparameter tuning using genetic algorithms. The primary tools include the :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` optimizer and the :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>`, which defines the hyperparameters to be tuned. This approach ensures efficient exploration of large parameter spaces, reducing the computational cost of hyperparameter search.

The :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` class is compatible with any model that adheres to the :class:`Estimator <sklearn.base.Estimator>` API in :mod:`scikit-learn`, making integration into existing workflows straightforward.

**Key Components**:

- **GeneticSearch**: The optimization engine, utilizing genetic algorithms to efficiently search for optimal hyperparameter configurations.
- **HyperparameterSpaceBuilder**: A builder class for defining both fixed and evolvable hyperparameters, supporting a variety of parameter types. This class provides a streamlined, user-friendly approach to constructing hyperparameter spaces.

Usage
-----

To get started with `mloptimizer`, follow these main steps:

1. **Define the Dataset**: Load or prepare your feature matrix (`X`) and target vector (`y`).
2. **Choose a Model**: Select a machine learning model that adheres to the :mod:`scikit-learn` API (e.g., :class:`XGBClassifier <xgboot.XGBClassifier>`).
3. **Set Up Hyperparameter Space**: Use :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>` to define the parameters you want to optimize, either by loading a default space or by adding custom hyperparameters.
4. **Run GeneticSearch**: Initialize :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` with the model and hyperparameter space, then call the `fit` method to start optimization.

**Using HyperparameterSpaceBuilder**:

The :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>` allows for a clean, structured setup of both fixed and evolvable hyperparameters:

- **Fixed Parameters**: Parameters that remain constant during the optimization process.
- **Evolvable Parameters**: Parameters that the genetic algorithm can adjust to find the best configuration.

You can add integer, float, and categorical parameters, or load default spaces for commonly used models.

Basic Usage Example
-------------------

The following example demonstrates the basic usage of :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` with :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>` to optimize an :class:`XGBClassifier <xgboot.XGBClassifier>` using default hyperparameters.

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from xgboost import XGBClassifier
    from sklearn.datasets import load_iris

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Define default hyperparameter space
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

    # Initialize GeneticSearch with default space
    opt = GeneticSearch(
        estimator_class=XGBClassifier,
        hyperparam_space=hyperparam_space,
        genetic_params_dict={"generations": 10, "population_size": 20}
    )

    # Run optimization
    opt.fit(X, y)
    print(opt.best_estimator_)

This setup leverages the default hyperparameter space for :class:`XGBClassifier <xgboot.XGBClassifier>` to start the optimization process immediately.

Custom HyperparameterSpace Example
----------------------------------

For specific tuning needs, use :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>` to define a custom hyperparameter space with both fixed and evolvable parameters. Here’s an example:

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from xgboost import XGBClassifier

    # Initialize HyperparameterSpaceBuilder
    builder = HyperparameterSpaceBuilder()

    # Add evolvable parameters
    builder.add_int_param("n_estimators", 50, 300)
    builder.add_float_param("learning_rate", 0.01, 0.3)
    builder.add_categorical_param("booster", ["gbtree", "dart"])

    # Set fixed parameters
    builder.set_fixed_param("max_depth", 5)

    # Build the custom hyperparameter space
    custom_hyperparam_space = builder.build()

    # Initialize GeneticSearch with custom space
    opt = GeneticSearch(
        estimator_class=XGBClassifier,
        hyperparam_space=custom_hyperparam_space,
        genetic_params_dict={"generations": 5, "population_size": 10}
    )

    # Run optimization
    opt.fit(X, y)

This example showcases adding custom integer, float, and categorical parameters, as well as fixed parameters to fine-tune the optimization process for :class:`XGBClassifier <xgboot.XGBClassifier>`.

Reproducibility
---------------

For consistent results, you can set a `seed` in :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>`. This ensures that repeated runs yield identical results, which is essential for experimental reproducibility.

Example:

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from xgboost import XGBClassifier
    from sklearn.datasets import load_iris

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Define default hyperparameter space
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

    # Initialize optimizer with a fixed seed
    opt = GeneticSearch(
        estimator_class=XGBClassifier,
        hyperparam_space=hyperparam_space,
        genetic_params_dict={"generations": 5, "population_size": 10},
        seed=42
    )

    # Run optimization
    opt.fit(X, y)

Setting the same `seed` value across multiple runs will produce identical results, enabling reliable comparison between experiments.

.. warning::

   On macOS with newer processor architectures (e.g., M1 or M2 chips), users may experience occasional reproducibility issues due to hardware-related differences in random number generation and floating-point calculations. To ensure consistency across runs, we recommend running `mloptimizer` within a Docker container configured for reproducible behavior. This approach helps isolate the environment and improves reproducibility on macOS hardware.
