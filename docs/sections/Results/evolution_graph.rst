Evolution Graph
===============

The evolution graph visualizes the progression of fitness scores across generations in a genetic optimization process. This plot helps you understand the algorithm’s convergence behavior, track improvements in fitness scores, and observe the distribution of individual scores within each generation.

Gallery Example
---------------
See the following example for practical usage and code details:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Example
     - Description
   * - :ref:`sphx_glr_auto_examples_plot_evolution.py`
     - Demonstrates how to set up and plot the evolution graph of a genetic algorithm optimization process.


Overview
--------

The evolution graph highlights key metrics throughout the optimization process.

The `mloptimizer` library provides a function to generate this graph using Plotly, making it interactive and customizable.

Saved Graph and Data Files
--------------------------

After the optimization completes, the evolution graph and related data are saved for future reference:

- **Graph Path**: An HTML file of the evolution graph is saved in the `graphics` directory.
- **Data Path**: CSV files with population data and logbook statistics are saved in the `results` directory.

For a detailed directory layout, refer to :doc:`directory_structure`.

**Note**: The evolution graph helps identify whether the genetic algorithm has converged or if additional generations might improve fitness further.
