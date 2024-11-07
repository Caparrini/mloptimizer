Defining Hyperparameter Spaces with HyperparameterSpaceBuilder
======================================================================

In this step, we will explore how to define hyperparameter spaces for optimization using :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>`. This builder allows you to create a comprehensive set of hyperparameters, including both fixed values and evolvable parameters, to maximize the flexibility and effectiveness of your optimization with :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>`.

Overview of HyperparameterSpaceBuilder
--------------------------------------

`HyperparameterSpaceBuilder` is a dedicated class for constructing hyperparameter spaces. It supports various data types, such as integers, floats, and categorical parameters, making it adaptable to a wide range of machine learning models. You can use it to define default spaces or create custom configurations tailored to your specific needs.

Key Methods
-----------

- **add_int_param(name, min_value, max_value)**: Adds an integer hyperparameter with a defined range to the space.
- **add_float_param(name, min_value, max_value, scale=100)**: Adds a float hyperparameter with a defined range and optional scaling.
- **add_categorical_param(name, values)**: Adds a categorical hyperparameter with a set of predefined values.
- **set_fixed_param(name, value)**: Sets a fixed parameter that will remain constant during optimization.
- **build()**: Builds and returns the completed hyperparameter space.
- **get_default_space(estimator_class)**: Loads a predefined hyperparameter space for the specified estimator.

Using Default Hyperparameter Spaces
-----------------------------------

`mloptimizer` provides predefined hyperparameter spaces for several common estimators, which can be loaded directly using the :meth:`get_default_space <mloptimizer.interfaces.HyperparameterSpaceBuilder.get_default_space>` method. This is particularly useful for rapid experimentation, as these default configurations are tuned for compatibility with various algorithms.

Supported estimators and their default hyperparameter spaces include:

- **Decision Tree Models**:
  - :class:`DecisionTreeClassifier <sklearn.tree.DecisionTreeClassifier>`
  - :class:`DecisionTreeRegressor <sklearn.tree.DecisionTreeRegressor>`
- **Random Forest Models**:
  - :class:`RandomForestClassifier <sklearn.ensemble.RandomForestClassifier>`
  - :class:`RandomForestRegressor <sklearn.ensemble.RandomForestRegressor>`
- **Extra Trees Models**:
  - :class:`ExtraTreesClassifier <sklearn.ensemble.ExtraTreesClassifier>`
  - :class:`ExtraTreesRegressor <sklearn.ensemble.ExtraTreesRegressor>`
- **Gradient Boosting Models**:
  - :class:`GradientBoostingClassifier <sklearn.ensemble.GradientBoostingClassifier>`
  - :class:`GradientBoostingRegressor <sklearn.ensemble.GradientBoostingRegressor>`
- **XGBoost Models**:
  - :class:`XGBClassifier <xgboost.XGBClassifier>`
  - :class:`XGBRegressor <xgboost.XGBRegressor>`
- **Support Vector Machines**:
  - :class:`SVC <sklearn.svm.SVC>`
  - :class:`SVR <sklearn.svm.SVR>`

These default hyperparameter spaces are available in JSON files within the repository, and `HyperparameterSpaceBuilder` can load them by referencing the relevant estimator class.

Example: **Loading a Default Hyperparameter Space**:

The example below demonstrates how to load a default hyperparameter space for :class:`XGBClassifier <xgboost.XGBClassifier>` using :meth:`get_default_space <mloptimizer.interfaces.HyperparameterSpaceBuilder.get_default_space>`.

.. code-block:: python

    from mloptimizer.interfaces import HyperparameterSpaceBuilder
    from xgboost import XGBClassifier

    # Load the default hyperparameter space for XGBClassifier
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

    print("Default hyperparameter space loaded for XGBClassifier.")

Creating a Custom Hyperparameter Space
--------------------------------------

In many cases, you may want to define a custom hyperparameter space with specific parameters tailored to your model. :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>` allows you to add parameters that :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` can evolve during optimization, as well as parameters with fixed values that will not change.

Example: **Adding Evolvable Parameters**:

The example below demonstrates how to use `HyperparameterSpaceBuilder` to add evolvable parameters (parameters that will be optimized) for a custom `XGBClassifier` configuration.

.. code-block:: python

    from mloptimizer.interfaces import HyperparameterSpaceBuilder

    # Initialize the builder
    builder = HyperparameterSpaceBuilder()

    # Add evolvable hyperparameters
    builder.add_int_param("n_estimators", 50, 300)
    builder.add_float_param("learning_rate", 0.01, 0.3)
    builder.add_categorical_param("booster", ["gbtree", "dart"])

    # Build the hyperparameter space
    custom_hyperparam_space = builder.build()

    print("Custom evolvable hyperparameter space created.")

Example: **Adding Fixed Parameters**:

You can also set fixed parameters that will remain constant during the optimization process. Here’s an example of setting both evolvable and fixed parameters.

.. code-block:: python

    # Set a fixed hyperparameter
    builder.set_fixed_param("max_depth", 5)

    # Add evolvable parameters
    builder.add_int_param("n_estimators", 100, 500)
    builder.add_float_param("subsample", 0.5, 1.0)

    # Build the custom hyperparameter space
    mixed_hyperparam_space = builder.build()

    print("Hyperparameter space with both fixed and evolvable parameters created.")

Saving and Loading Hyperparameter Spaces
----------------------------------------

:class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>` also provides functionality to save and load hyperparameter spaces, allowing you to reuse configurations across projects or experiments.

Example: **Saving and Loading a Hyperparameter Space**:

.. code-block:: python

    # Save the custom hyperparameter space to a file
    builder.save_space(mixed_hyperparam_space, "custom_hyperparam_space.json", overwrite=True)

    # Load the saved hyperparameter space
    loaded_hyperparam_space = builder.get_default_space(XGBClassifier)

    print("Hyperparameter space saved and reloaded.")

Summary
-------

In this step, we covered:

1. Defining hyperparameter spaces using :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>`.
2. Creating both fixed and evolvable parameters for flexible optimization.
3. Loading default hyperparameter spaces for supported models.
4. Saving and loading hyperparameter spaces for easy reuse.

Once your hyperparameter space is defined, you’re ready to move on to Step 3, where we’ll execute and monitor the optimization process.
