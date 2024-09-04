"""
Regression Example
==================
This example shows how to optimize a regression model using the mloptimizer library.
"""

# %%
# Load the diabetes dataset
# -------------------------
# The diabetes dataset is used to demonstrate the optimization of a regression model.
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

# %%
# The dataset has 10 features, but we will use only one for this plot (the Body Mass Index (BMI)).
import matplotlib.pyplot as plt
plt.scatter(X[:,2], y)
plt.xlabel("Body Mass Index")
plt.ylabel("Disease Progression")
plt.title("Diabetes Dataset")
plt.show()


# %%
# Default hyperparameter space
# ----------------------------
# Define the HyperparameterSpace, you can use the default hyperparameters for the machine learning model
# that you want to optimize. In this case we use the default hyperparameters for a RandomForestRegressor.
from mloptimizer.hyperparams import HyperparameterSpace
from sklearn.ensemble import RandomForestRegressor

hyperparam_space = HyperparameterSpace.get_default_hyperparameter_space(RandomForestRegressor)

# %%
# We use the Optimizer class to optimize a regression model.
from mloptimizer.core import Optimizer
mlopt = Optimizer(estimator_class=RandomForestRegressor,
                      hyperparam_space=hyperparam_space,
                      features=X, labels=y, folder="regression_example")

clf = mlopt.optimize_clf(5, 5)

# %%
# The best individual is returned by the optimize_clf method.
# However, this individual is not the trained model, but the hyperparameters used to train the best model.
# As the literature suggests, we can use the best hyperparameters to train the model again with all the data
# to obtain the best model or use a cross-validation estimator to make new predictions.
print(clf)

# %%
# Custom hyperparameter space
# ---------------------------
# The hyperparameter space can be defined by the user.
# In this example, we define a new hyperparameter space for the RandomForestRegressor model.
from mloptimizer.hyperparams import Hyperparam
fixed_hyperparams = {'n_estimators': 100}
evolvable_hyperparams = {'max_depth': Hyperparam(name='max_depth', min_value=1,
                                                 max_value=10, hyperparam_type='int'),
                         'min_samples_split': Hyperparam(name='min_samples_split', min_value=2,
                                                         max_value=10, hyperparam_type='int'),
                         'min_samples_leaf': Hyperparam(name='min_samples_leaf', min_value=1,
                                                        max_value=10, hyperparam_type='int')}
custom_hyperparam_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)


# %%
# Furthermore, it is possible to set the metrics used to evaluate the model, and use one of them as the fitness score.
from sklearn.metrics import mean_squared_error, root_mean_squared_error
regression_metrics = {
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error
}

mlopt = Optimizer(estimator_class=RandomForestRegressor,
                  hyperparam_space=custom_hyperparam_space,
                  fitness_score='rmse', metrics=regression_metrics,
                  features=X, labels=y, folder="regression_example")

clf = mlopt.optimize_clf(5, 5)

print(clf)