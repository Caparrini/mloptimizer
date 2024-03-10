"""
Quickstart example
====================================
Quick example of use of the library to optimize a decision tree classifier.
Firstly, we import the necessary libraries to get data and plot the results.
"""

from mloptimizer.core import SklearnOptimizer
from mloptimizer.hyperparams import HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
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
# The TreeOptimizer class is the main wrapper for the optimization of a Decision Tree Classifier.
# The first argument is the vector of features,
# the second is the vector of labels and
# the third (if provided) is the name of the folder where the results of mloptimizer Optimizers are saved.
# The default value for this folder is "Optimizer"
opt = SklearnOptimizer(clf_class=DecisionTreeClassifier, features=X, labels=y,
                       hyperparam_space=hyperparam_space, folder="Optimizer")

# %%
# To optimizer the classifier we need to call the optimize_clf method.
# The first argument is the number of generations and
# the second is the number of individuals in each generation.
clf = opt.optimize_clf(10, 10)

# %%
# The structure of the Optimizer folder is as follows:

del opt
