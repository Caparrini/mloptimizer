"""
Evolution (logbook) graph
====================================
mloptimizer provides a function to plot the evolution of the fitness function.
"""

from sklearn.tree import DecisionTreeClassifier
from mloptimizer.application.reporting.plots import plotly_logbook
from mloptimizer.domain.evaluation import kfold_stratified_score
import plotly
import os
from sklearn.datasets import load_iris
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch

# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# Define the HyperparameterSpace, you can use the default hyperparameters for the machine learning model
# that you want to optimize. In this case we use the default hyperparameters for a DecisionTreeClassifier.
# Another dataset or a custom one can be used
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(estimator_class=DecisionTreeClassifier)

# %%
# The GeneticSearch class is the main wrapper for the optimization of a machine learning model.
opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
        genetic_params_dict={"generations": 30, "population_size": 100},
        eval_function=kfold_stratified_score,
    )

# %%
# To optimizer the classifier we need to call the fit method.
opt.fit(X, y)


# %%
# We can plot the evolution of the fitness function.
# The black lines represent the max and min fitness values across all generations.
# The green, red and blue line are respectively the max, min and avg fitness value for each generation.
# Each grey point in the graph represents an individual.
population_df = opt.optimizer_service.optimizer.genetic_algorithm.population_2_df()
g_logbook = plotly_logbook(opt.optimizer_service.optimizer.genetic_algorithm.logbook, population_df)
plotly.io.show(g_logbook)

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt.optimizer_service.optimizer.tracker.graphics_path)
print(os.listdir(opt.optimizer_service.optimizer.tracker.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt.optimizer_service.optimizer.tracker.results_path)
print(os.listdir(opt.optimizer_service.optimizer.tracker.results_path))

del opt
