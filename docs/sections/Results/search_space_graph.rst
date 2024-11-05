Search Space Graph
==================

The search space graph provides a visualization of the hyperparameter values explored during the genetic optimization process. This plot reveals the distribution of evaluated hyperparameters and helps you identify the value ranges that achieved higher fitness scores, offering valuable insights into the search space landscape and parameter sensitivity.

Gallery Example
---------------

Refer to the following example for practical usage and code details:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Example
     - Description
   * - :ref:`sphx_glr_auto_examples_plot_search_space.py`
     - Demonstrates how to set up and plot the search space graph for a genetic algorithm optimization process.


Understanding the Search Space Graph
------------------------------------

The search space graph is a scatter plot that visually represents the hyperparameter configurations tested by the genetic algorithm:

- **Axes**: Each axis represents an evolvable hyperparameter. This allows you to observe the ranges of values explored during the optimization process.
- **Points**: Each point in the graph corresponds to a unique combination of hyperparameters, with its position representing specific values for each parameter.
- **Fitness Scores**: Points are typically color-coded or sized based on fitness scores, highlighting which parameter combinations yielded the best results.

This visualization helps you identify hyperparameter ranges associated with higher fitness scores, providing insights into parameter sensitivity and guiding future optimizations.

How to Use This Graph
---------------------

The search space graph is especially useful for assessing:

- **Parameter Sensitivity**: By observing clusters of high-fitness points, you can identify hyperparameters that significantly impact model performance.
- **Value Ranges for Further Tuning**: If certain parameter ranges are associated with better fitness, you can refine future optimization runs to focus on those areas.
- **Relationships Between Parameters**: The graph can reveal interactions between hyperparameters, such as values that consistently lead to higher fitness when used in combination.

Saved Graph and Data Files
--------------------------

After the optimization completes, the search space graph and related data are saved for future reference:

- **Graph Path**: An HTML file of the search space graph is saved in the `graphics` directory.
- **Data Path**: CSV files with population data, including hyperparameter values and fitness scores, are saved in the `results` directory.

For a detailed directory layout, refer to :doc:`directory_structure`.

**Takeaway**: The search space graph provides valuable insights into hyperparameter effectiveness and sensitivity, allowing you to identify which parameters significantly impact performance and explore promising value ranges for further tuning.
