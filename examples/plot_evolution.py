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
from sklearn.model_selection import StratifiedKFold

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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
        **{"generations": 30, "population_size": 100},
        # eval_function=kfold_stratified_score, # Deprecated
        cv=cv
    )

# %%
# To optimizer the classifier we need to call the fit method.
opt.fit(X, y)


# %%
# We can plot the evolution of the fitness function.
population_df = opt.populations_
g_logbook = plotly_logbook(opt.logbook_, population_df)
plotly.io.show(g_logbook)

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt._optimizer_service.optimizer.tracker.graphics_path)
print(os.listdir(opt._optimizer_service.optimizer.tracker.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt._optimizer_service.optimizer.tracker.results_path)
print(os.listdir(opt._optimizer_service.optimizer.tracker.results_path))

del opt
