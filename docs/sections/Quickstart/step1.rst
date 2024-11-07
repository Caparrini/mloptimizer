Setting Up an Optimization with GeneticSearch
=====================================================

In this step, you’ll learn how to set up `GeneticSearch` as an optimizer for your machine learning model, using it similarly to `GridSearchCV` in :mod:`scikit-learn`. `GeneticSearch` is compatible with any model that adheres to the :class:`Estimator <sklearn.base.Estimator>` API, making it easy to integrate into pipelines. This guide covers initializing `GeneticSearch`, configuring key parameters, and using it to optimize model hyperparameters efficiently.

Overview of GeneticSearch
-------------------------

`GeneticSearch` is an optimization class based on genetic algorithms, a powerful search technique that reduces search time by iteratively refining solutions. By treating each set of hyperparameters as an “individual” in a population, `GeneticSearch` evolves the population over multiple generations to find the optimal configuration. This approach is particularly useful for large or complex search spaces where traditional grid or random search would be too computationally expensive.

Configuring Genetic Parameters
------------------------------

The `genetic_params_dict` allows you to control how `GeneticSearch` performs the optimization. By default, the following parameters are set but can be customized to refine the genetic algorithm’s behavior. These parameters reference DEAP’s genetic algorithm operations and include:

- **generations**: Number of evolutionary cycles the genetic algorithm will run. More generations allow for deeper refinement but increase computational time.
- **population_size**: Number of individuals in each generation. Larger populations explore the search space more thoroughly but require more computational resources.
- **cxpb** (crossover probability): The probability of crossover, where two individuals combine to create offspring. Higher values promote diversity in the population.
- **mutpb** (mutation probability): Probability of mutation, introducing random changes in individuals. Higher values increase diversity but may slow convergence.
- **n_elites**: Number of top-performing individuals to retain in each generation. This keeps the best-performing solutions in the population, aiding stability.
- **tournsize**: Tournament size, which controls the selection pressure. Larger values make selection more competitive, favoring fitter individuals.
- **indpb**: Independent probability of mutating each attribute. This fine-tunes the mutation process, with higher values leading to more exploratory changes.

**Custom Genetic Parameters**:

The example below demonstrates how to override the default `genetic_params_dict` with specific values for `generations`, `population_size`, `cxpb`, `mutpb`, and other parameters to control the optimization behavior.

.. code-block:: python

    # Initialize GeneticSearch with custom genetic parameters
    opt = GeneticSearch(
        estimator_class=XGBClassifier,
        hyperparam_space=hyperparam_space,
        genetic_params_dict={
            "generations": 20,
            "population_size": 30,
            "cxpb": 0.7,
            "mutpb": 0.4,
            "n_elites": 5,
            "tournsize": 4,
            "indpb": 0.1
        },
        seed=42  # Set seed for reproducibility
    )

    print("GeneticSearch initialized with custom genetic parameters.")

Basic Initialization
--------------------

Here’s how to initialize `GeneticSearch` in a simple example:

1. **Load your data**: Start by loading or preparing your dataset, ensuring you have features (`X`) and labels (`y`).
2. **Define the model and hyperparameter space**: Choose a model (e.g., :class:`XGBClassifier <xgboot.XGBClassifier>`) and set up a hyperparameter space with `HyperparameterSpaceBuilder`.
3. **Initialize GeneticSearch**: Use the chosen model, hyperparameter space, and genetic algorithm parameters to set up `GeneticSearch`.

### Example: Setting Up GeneticSearch with Default Parameters

This example demonstrates using `GeneticSearch` to optimize an `XGBClassifier` with default parameters.

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from xgboost import XGBClassifier
    from sklearn.datasets import load_iris

    # 1) Load the dataset
    X, y = load_iris(return_X_y=True)

    # 2) Define the hyperparameter space (using default space for XGBClassifier)
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

    # 3) Initialize GeneticSearch
    opt = GeneticSearch(
        estimator_class=XGBClassifier,
        hyperparam_space=hyperparam_space,
        genetic_params_dict={"generations": 10, "population_size": 20}
    )

    # Ready to run optimization in the next step
    print("GeneticSearch initialized and ready for optimization.")

Incorporating GeneticSearch into Pipelines
------------------------------------------

One of the benefits of `GeneticSearch` is that it can be treated similarly to `GridSearchCV`, enabling integration into `scikit-learn` pipelines. Here’s an example using `Pipeline` to chain data preprocessing and model optimization.

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Define a preprocessing and optimization pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),               # Standardize features
        ("genetic_search", GeneticSearch(
            estimator_class=XGBClassifier,
            hyperparam_space=hyperparam_space,
            genetic_params_dict={"generations": 10, "population_size": 20},
            seed=42
        ))
    ])

    # Fit pipeline on the dataset
    pipeline.fit(X, y)

    print("Pipeline with GeneticSearch completed.")

This example shows how to integrate `GeneticSearch` with other preprocessing steps in a pipeline, treating it as you would any other estimator in :mod:`scikit-learn`.

Summary
-------

In this step, you learned to:

1. Initialize `GeneticSearch` with a compatible model and hyperparameter space.
2. Configure essential genetic algorithm parameters to control the search process.
3. Incorporate `GeneticSearch` into a machine learning pipeline for seamless optimization.

Once `GeneticSearch` is set up, you’re ready to define your hyperparameter space in Step 2, fine-tuning the search space to suit your model’s needs.
