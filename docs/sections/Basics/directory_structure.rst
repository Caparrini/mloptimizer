=============================
Optimizer Directory Structure
=============================

When an optimizer is run, it generates a directory in the current working directory (or the given directory as input).
This directory, named in the format `YYYYMMDD_nnnnnnnnnn_OptimizerName`, contains the results of the optimization process,
including the best estimator found, a log file detailing the optimization steps, and the final result of the optimization.

Directory Structure
-------------------
The directory structure is as follows:

.. code-block:: bash

    ├── checkpoints
    │   ├── cp_gen_0.pkl
    │   └── cp_gen_1.pkl
    ├── graphics
    │   ├── logbook.html
    │   └── search_space.html
    ├── opt.log
    ├── progress
    │   ├── Generation_0.csv
    │   └── Generation_1.csv
    └── results
        ├── logbook.csv
        └── populations.csv

Directory Contents
------------------
Each item in the directory serves a specific purpose:

- `checkpoints`: Contains the checkpoint files for each generation of the genetic optimization process. These files preserve the state of the optimization process at each generation, enabling the process to be resumed from a specific point if necessary.
    - `cp_gen_0.pkl`, `cp_gen_1.pkl`: These are the individual checkpoint files for each generation. They are named according to the generation number and are saved in Python's pickle format.

- `graphics`: Contains HTML files for visualizing the optimization process.
    - `logbook.html`: Provides a graphical representation of the logbook, which records the statistics of the optimization process over generations.
    - `search_space.html`: Provides a graphical representation of the search space of the optimization process.

- `opt.log`: The log file for the optimization process. It contains detailed logs of the optimization process, including the performance of the algorithm at each generation.

- `progress`: Contains CSV files that record the progress of the optimization process for each generation.
    - `Generation_0.csv`, `Generation_1.csv`: These are the individual progress files for each generation. They contain detailed information about each individual in the population at each generation.

- `results`: Contains CSV files with the results of the optimization process.
    - `logbook.csv`: This file is a CSV representation of the logbook, which records the statistics of the optimization process over generations.
    - `populations.csv`: This file contains the final populations of the optimization process. It includes the hyperparameters and fitness values of each individual in the population.
