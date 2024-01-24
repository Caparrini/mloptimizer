============
Introduction
============
This user guide is an introduction to the MLOptimizer library,
designed to optimize machine learning models with a focus on ease of use of the Deap library.
The guide will demonstrate the library's capabilities through examples and highlight its features and customization options.

MLOptimizer is intended to complement detailed API documentation, offering practical insights and optimal usage strategies.

While MLOptimizer integrates seamlessly with Python's machine learning ecosystem,
it's built on Deap optimization algorithms, which are not specific to machine learning.
This guide primarily uses Python examples, providing a
straightforward path for practitioners familiar with Python-based machine learning libraries.

Features
--------
The goal of mloptimizer is to provide a user-friendly, yet powerful optimization tool that:

- Easy to use
- DEAP-based genetic algorithm ready to use with several machine learning algorithms
- Default hyperparameter ranges
- Default score functions for evaluating the performance of the model
- Reproducibility of results
- Extensible with more machine learning algorithms that comply with the Scikit-Learn API
- Customizable hyperparameter ranges
- Customizable score functions


Using mloptimizer
-----------------

Step 1: Select and Setup the Algorithm to Optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MLOptimizer uses a wrapper for the algorithm for classification or regression that is going to be optimized.
It currently supports the following wrappers and their corresponding algorithms:

- `TreeOptimizer`: Decision Tree Classifier from scikit-learn
- `RandomForestOptimizer`: Random Forest Classifier from scikit-learn
- `ExtraTreesOptimizer`: Extra Trees Classifier from scikit-learn
- `GradientBoostingOptimizer`: Gradient Boosting Classifier from scikit-learn
- `SVCClassifierOptimizer`: Support Vector Classifier from scikit-learn
- `KerasClassifierOptimizer`: Keras Classifier
- `XGBClassifierOptimizer`: XGBoost Classifier

Let’s assume that we want to fine-tune the decision tree classifier from scikit-learn, wrapped in `TreeOptimizer`.

To instantiate the wrapper, you need to specify the dataset to work with, the input features (as a matrix),
and the output features (as a column).

The wrapper has a variable with the set of hyperparameters to be explored.
For the case of the decision tree classifier in `TreeOptimizer`,
the default hyperparameters and their exploration ranges are:

- `min_samples_split`, range [2, 50]
- `min_samples_leaf`, range [1, 20]
- `max_depth`, range [2, 20]
- `min_impurity_decrease`, range [0, 0.15] in 1000 steps
- `ccp_alpha`, range [0, 0.003] in 100,000 steps

For a quick start, we will explore the default hyperparameters using a default range for exploring each of them.

Similarly, in the wrapper, you can set up the metric to be optimized
(the parameter is called `score_function` and the default value is accuracy)
from the metrics available in scikit-learn (`sklearn.metrics`)
and the evaluation setting (the parameter is called `model_evaluation`
and the default value is the `train_score`).

See the API reference for more details on setting up the wrapper and the optimization.

Step 2: Running the Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you have instantiated the wrapper with the algorithm to optimize, you can run the genetic optimization.

Typically, you should set the number of generations (3 by default) and the size of the population (10 by default).

The optimization returns the best classifier found during the genetic optimization,
tuned with the corresponding hyperparameters. Additionally, during the optimization, a structure of
directories is created to store the results of the optimization process. The structure of the directories is
explained in the section on the optimizer output directory structure and contain useful information, logs,
checkpoints and plots.

Step 3: Using the Outcome of the Optimization Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The result of the optimization process is the optimal classifier object.
You can use this object to make predictions on your dataset.
For example, if `clf_result` is the returned classifier, you can use `clf_result.predict(X)` to make predictions.

In addition to the optimal classifier,
you can explore the outcomes of the optimization process,
such as the evolution of the population and the best score at each generation.
These outcomes are stored in the directory created by the optimizer,
as explained in the section on the optimizer output directory structure.

.. warning::
   mloptimizer is not a machine learning library. It is a hyperparameter optimization library that can be used with any machine learning library that complies with the scikit-learn API.

.. warning::
   Before optimizing a machine learning model using mloptimizer it is recommended first to have a cleaned dataset. mloptimizer does not provide any data preprocessing or cleaning tools.

.. note::
   The examples in this guide are aligned with the latest version of mloptimizer. Users are encouraged to ensure they are using the most recent release to fully leverage the library's capabilities.
