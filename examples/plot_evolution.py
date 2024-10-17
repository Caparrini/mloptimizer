"""
Evolution (logbook) graph
====================================
mloptimizer provides a function to plot the evolution of the fitness function.
"""

from mloptimizer.application import Optimizer
from mloptimizer.domain.hyperspace import HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from mloptimizer.application.reporting.plots import plotly_logbook
from mloptimizer.domain.evaluation import kfold_stratified_score
import plotly
import os
from sklearn.datasets import load_iris

# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# Define the HyperparameterSpace, you can use the default hyperparameters for the machine learning model
# that you want to optimize. In this case we use the default hyperparameters for a DecisionTreeClassifier.
# Another dataset or a custom one can be used
hyperparam_space = HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier)

# %%
# We use the default TreeOptimizer class to optimize a decision tree classifier.
opt = Optimizer(estimator_class=DecisionTreeClassifier, features=X, labels=y,
                hyperparam_space=hyperparam_space, eval_function=kfold_stratified_score,
                folder="Evolution_example")

# %%
# To optimizer the classifier we need to call the optimize_clf method.
# The first argument is the number of generations and
# the second is the number of individuals in each generation.
clf = opt.optimize_clf(100, 30)


# %%
# We can plot the evolution of the fitness function.
# The black lines represent the max and min fitness values across all generations.
# The green, red and blue line are respectively the max, min and avg fitness value for each generation.
# Each grey point in the graph represents an individual.
population_df = opt.runs[-1].population_2_df()
g_logbook = plotly_logbook(opt.logbook, population_df)
plotly.io.show(g_logbook)

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt.tracker.graphics_path)
print(os.listdir(opt.tracker.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt.tracker.results_path)
print(os.listdir(opt.tracker.results_path))

del opt
