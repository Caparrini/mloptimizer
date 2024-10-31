"""
Search space graph
====================================
mloptimizer provides a function to plot the search space of the optimization.
"""

from sklearn.tree import DecisionTreeClassifier
from mloptimizer.domain.evaluation import kfold_stratified_score
from mloptimizer.application.reporting.plots import plotly_search_space
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
# The GeneticSearch class is used to optimize the hyperparameters of a machine learning model.
opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=hyperparam_space,
        genetic_params_dict={"generations": 30, "population_size": 100},
        eval_function=kfold_stratified_score
        )

# %%
# To optimizer the classifier we need to call the fit method.
opt.fit(X, y)
clf = opt.best_estimator_


# %%
# Following we can generate the plot of the search space
population_df = opt.optimizer_service.optimizer.genetic_algorithm.population_2_df()
param_names = list(opt.optimizer_service.hyperparam_space.evolvable_hyperparams.keys())
param_names.append("fitness")
df = population_df[param_names]
g_search_space = plotly_search_space(df, param_names)
plotly.io.show(g_search_space)

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt.optimizer_service.optimizer.tracker.graphics_path)
print(os.listdir(opt.optimizer_service.optimizer.tracker.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt.optimizer_service.optimizer.tracker.results_path)
print(os.listdir(opt.optimizer_service.optimizer.tracker.results_path))

del opt
