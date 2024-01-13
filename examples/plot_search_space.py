"""
Search space graph
====================================
mloptimizer provides a function to plot the search space of the optimization.
"""

from mloptimizer.genoptimizer import TreeOptimizer
from mloptimizer.plots import plotly_search_space
import plotly
import os
from sklearn.datasets import load_iris

# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# We use the default TreeOptimizer class to optimize a decision tree classifier.
opt = TreeOptimizer(X, y, "Search_space_example")

# %%
# To optimizer the classifier we need to call the optimize_clf method.
# The first argument is the number of generations and
# the second is the number of individuals in each generation.
clf = opt.optimize_clf(10, 10)


# %%
# Following we can generate the plot of the search space
population_df = opt.population_2_df()
param_names = list(opt.get_hyperparams().keys())
param_names.append("fitness")
df = population_df[param_names]
g_search_space = plotly_search_space(df, param_names)
plotly.io.show(g_search_space)

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt.graphics_path)
print(os.listdir(opt.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt.results_path)
print(os.listdir(opt.results_path))

del opt
