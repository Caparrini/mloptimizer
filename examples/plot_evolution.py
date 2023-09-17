"""
Evolution (logbook) graph
====================================
mloptimizer provides a function to plot the evolution of the fitness function.
"""

from mloptimizer.genoptimizer import TreeOptimizer
from mloptimizer.plots import plotly_logbook
import plotly
import os
from sklearn.datasets import load_iris

# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# We use the default TreeOptimizer class to optimize a decision tree classifier.
opt = TreeOptimizer(X, y, "Evolution_example")

# %%
# To optimizer the classifier we need to call the optimize_clf method.
# The first argument is the number of generations and
# the second is the number of individuals in each generation.
clf = opt.optimize_clf(10, 10)


# %%
# We can plot the evolution of the fitness function.
# The black lines represent the max and min fitness values across all generations.
# The green, red and blue line are respectively the max, min and avg fitness value for each generation.
# Each grey point in the graph represents an individual.
population_df = opt.population_2_df()
g_logbook = plotly_logbook(opt.logbook, population_df)
plotly.io.show(g_logbook)

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt.graphics_path)
print(os.listdir(opt.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt.results_path)
print(os.listdir(opt.results_path))

del opt
