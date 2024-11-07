Reviewing and Interpreting Results
==========================================

After completing optimization with :class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>`, it’s essential to interpret the outcomes to understand the best-performing model, its hyperparameters, and the optimization’s overall effectiveness. This step will guide you through accessing the best model, understanding cross-validation vs. full-dataset training, interpreting key metrics, and visualizing the optimization process.

Accessing the Best Model and Parameters
----------------------------------------

Once optimization completes, `GeneticSearch` provides access to the best model and hyperparameters found. The best model is stored as `opt.best_estimator_`, containing the optimal hyperparameters identified by the algorithm.

**Note**: The model in `best_estimator_` (and the one returned by `fit()`) is retrained on the entire dataset (`X` and `y`) provided to the optimization, regardless of the cross-validation method used during the search. During optimization, cross-validation is used only to assess candidate models’ fitness scores; the final model is fully trained to ensure the best performance on all available data.

**Example: Accessing and Displaying the Best Model**

.. code-block:: python

    # Display the best estimator found during optimization
    print("Best Estimator:")
    print(opt.best_estimator_)

The output will show the details of the best model, including its class and the optimized hyperparameters:

.. code-block:: text

    DecisionTreeClassifier(ccp_alpha=0.00055, max_depth=4,
                           min_impurity_decrease=0.001, min_samples_split=5,
                           random_state=296596)

**Example: Accessing Best Hyperparameters and Fitness Score**

The best hyperparameters and fitness score can be accessed directly through `best_params_` and `best_fitness_` attributes.

.. code-block:: python

    # Display the best hyperparameters and fitness score
    print("Best Hyperparameters:", opt.best_params_)
    print("Best Fitness Score:", opt.best_fitness_)

These attributes provide a summary of the best configuration found and its performance score, allowing you to assess the quality of the optimized model.

**Using the Best Model for Prediction or Scoring**

You can now use the `best_estimator_` model to make predictions or evaluate its performance on new data. For example:

.. code-block:: python

    # Using the best model to make predictions
    y_pred = opt.best_estimator_.predict(X_test)

    # Evaluating its performance on a test set
    score = opt.best_estimator_.score(X_test, y_test)
    print("Test Set Score:", score)

Key Considerations for Result Interpretation
--------------------------------------------

When reviewing results, consider the following:

- **Convergence Behavior**: If the evolution graph shows that fitness scores stabilized, the algorithm likely found an optimal solution. If scores continued improving, additional generations may yield further gains.
- **Parameter Sensitivity**: The search space graph can help identify parameters that had a strong impact on fitness. Hyperparameters with a narrower range near high fitness scores are likely more sensitive.
- **Validation**: For a comprehensive performance assessment, evaluate the `best_estimator_` on a separate validation or test set if available. This provides an unbiased measure of its effectiveness on new data.
- **Generalizability**: If you plan to use this model for similar tasks, the best hyperparameters identified can serve as a strong starting point for future optimizations.

Visualizing Optimization Results
--------------------------------

To gain insights into the optimization process, you can visualize the fitness evolution over generations and the search space explored by the genetic algorithm. `mloptimizer` includes built-in functions to generate these plots.

### Evolution (Logbook) Graph

The evolution graph displays the fitness function’s progress across generations, showing the maximum, minimum, and average fitness values for each generation. This visualization helps you understand the convergence pattern and whether the optimization reached a stable solution.

**Example: Generating the Evolution Graph**

.. code-block:: python

    from mloptimizer.application.reporting.plots import plotly_logbook
    import plotly.io as pio

    # Plot the evolution graph
    population_df = opt.populations_
    evolution_graph = plotly_logbook(opt.logbook_, population_df)
    pio.show(evolution_graph)

In this graph:
- **Black lines** represent the max and min fitness values across generations.
- **Green, red, and blue lines** correspond to the max, min, and average fitness values per generation.
- **Gray points** indicate individual fitness values within each generation, providing a sense of population diversity.

At the end of the optimization, the evolution graph is saved as an HTML file for easy reference. For the location of the saved plot, refer to the results folder’s structure in the documentation: :doc:`../Results/directory_structure`.

### Search Space Graph

The search space graph visualizes the hyperparameter values explored by the genetic algorithm. This plot shows the range of values tested for each hyperparameter and highlights the fitness scores associated with each combination, providing insight into the hyperparameter landscape.

**Example: Generating the Search Space Graph**

.. code-block:: python

    from mloptimizer.application.reporting.plots import plotly_search_space

    # Get population data and relevant parameters
    population_df = opt.populations_
    param_names = list(opt.get_evolvable_hyperparams().keys())
    param_names.append("fitness")

    # Create the search space plot
    search_space_graph = plotly_search_space(population_df[param_names], param_names)
    pio.show(search_space_graph)

In the search space graph:
- Each point represents a unique hyperparameter configuration tested by the genetic algorithm.
- The distribution of points shows the explored search space, helping you identify which hyperparameter ranges yielded higher fitness scores.

Results and Directory Structure
-------------------------------

After optimization completes, `GeneticSearch` generates a results folder containing detailed information about the best model and other optimization data. This folder includes:

- **Best Model Details**: Information on the best-performing model and its hyperparameters.
- **Evolution Log**: Data on fitness scores and hyperparameter values for each generation.
- **Saved Visualizations**: HTML files for the evolution and search space graphs.

For more details on the results folder structure, refer to the documentation: :doc:`../Results/directory_structure`.

Summary
-------

In this final step, we covered:

1. Accessing the best model and interpreting its hyperparameters and fitness score.
2. Using the best model for predictions or scoring on test data.
3. Visualizing the optimization process using evolution and search space graphs.
4. Understanding and interpreting optimization trends and parameter sensitivity.

This concludes the Quick Start guide. You’re now equipped to optimize hyperparameters using `GeneticSearch` and interpret the outcomes effectively, enabling you to fine-tune models for improved performance on your tasks.
