Running and Monitoring Optimization
===========================================

In this step, we cover how to execute the optimization process with :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>` and monitor key metrics. The progress output from `GeneticSearch` provides real-time feedback, allowing you to track the optimization’s performance across generations and assess convergence.

Executing the Optimization
--------------------------

Once you have defined your model and hyperparameter space with :class:`HyperparameterSpaceBuilder <mloptimizer.interfaces.HyperparameterSpaceBuilder>`, you’re ready to execute the optimization with `GeneticSearch`.

To start the optimization, call the `fit` method on your `GeneticSearch` instance, passing in the feature matrix (`X`) and target vector (`y`). This initiates the genetic algorithm, which runs the optimization over the specified number of generations and population size, iteratively refining hyperparameters.

**Example: Running GeneticSearch**

The example below demonstrates running `GeneticSearch` with a basic configuration:

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from xgboost import XGBClassifier
    from sklearn.datasets import load_iris

    # Load the dataset
    X, y = load_iris(return_X_y=True)

    # Define the hyperparameter space using the default space for XGBClassifier
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

    # Initialize GeneticSearch
    opt = GeneticSearch(
        estimator_class=XGBClassifier,
        hyperparam_space=hyperparam_space,
        genetic_params_dict={"generations": 10, "population_size": 20},
        seed=42
    )

    # Execute the optimization
    opt.fit(X, y)

    print("Optimization completed. Best estimator found.")

Progress Monitoring Output
--------------------------

During the optimization process, `GeneticSearch` provides real-time feedback on the console, updating progress with each generation. The typical output includes information about:

- **Progress Percentage**: Displays the percentage of generations completed out of the total specified.
- **Best Fitness**: Shows the highest score (fitness) achieved so far, reflecting the performance of the best hyperparameter set found by the algorithm.
- **Generation Speed**: Indicates the processing rate (e.g., iterations per second) to give an estimate of runtime.

**Example Output**

Here’s a sample output from `GeneticSearch` while running the `fit()` method:

.. code-block:: text

    WARNING:root:The folder . already exists and it will be used
    INFO:mloptimizer.log:Initiating genetic optimization...
    INFO:mloptimizer.log:Algorithm: Optimizer

    Genetic execution:   0%|          | 0/31 [00:00<?, ?it/s, best fitness=?]
    Genetic execution:   3%|▎         | 1/31 [00:00<00:00, 134.70it/s, best fitness=0.96]
    Genetic execution:  10%|▉         | 3/31 [00:00<00:01, 20.15it/s, best fitness=0.98]
    Genetic execution:  19%|█▉        | 6/31 [00:00<00:01, 17.76it/s, best fitness=0.98]
    Genetic execution:  29%|██▉       | 9/31 [00:00<00:01, 16.89it/s, best fitness=0.987]
    ...
    Genetic execution: 100%|██████████| 31/31 [00:02<00:00, 13.83it/s, best fitness=0.987]

    DecisionTreeClassifier(ccp_alpha=0.00055, max_depth=4,
                           min_impurity_decrease=0.001, min_samples_split=5,
                           random_state=296596)

Each line updates in real-time, providing:

- **Progress Bar**: A visual representation of the optimization's completion percentage.
- **Current Generation**: Tracks the iteration count out of the total specified (e.g., `31/31`).
- **Best Fitness**: Displays the highest fitness score achieved up to that generation, helping you monitor improvements over time.

.. note::

   The `fit()` method produces continuous updates on each generation’s progress. This is helpful for tracking long-running optimizations and observing whether fitness scores are converging.

Interpreting Progress and Convergence
--------------------------------------

As the optimization proceeds, you can interpret the progress output to assess how well the algorithm is performing:

- **Increasing Best Fitness**: A consistently improving best fitness score indicates that the optimization is effectively exploring and improving configurations. If the best fitness stabilizes, this may signal convergence.
- **Generation Speed**: The time per generation (iterations per second, or `it/s`) gives an estimate of runtime. If the rate drops significantly, it might indicate an increased computational load or a larger population size affecting speed.

Results and Directory Structure
-------------------------------

After the optimization completes, `GeneticSearch` generates a results folder containing details about the best model found and related optimization data. For more information on the folder structure and the types of files saved, refer to the `Directory Structure` section in the documentation: :doc:`../Results/directory_structure`.

Summary
-------

In this step, we covered:

1. Running the optimization with `GeneticSearch` using the `fit` method.
2. Monitoring key metrics during optimization, including best fitness, generation progress, and iteration speed.
3. Reviewing the generated results and directory structure for detailed output information.

With the optimization complete, proceed to Step 4 to review and interpret the final results, identifying the best model and its parameters.
