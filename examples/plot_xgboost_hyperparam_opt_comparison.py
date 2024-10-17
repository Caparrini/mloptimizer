"""
XGBoost - Genetic vs Grid Search vs Random Search vs Bayesian Optimization
==========================================================================
mloptimizer example optimization of iris dataset comparing hyperparameter tuning techniques:

1) Genetic optimization - mloptimizer
2) Grid Search - scikit-learn
3) Random Search - scikit-learn
4) Bayesian Optimization - hyperopt
"""

# %%
# Imports
# -------
# The necessary libraries for the example are imported.

import pandas as pd
import numpy as np
from time import time
from functools import reduce
import plotly

from mloptimizer.application import Optimizer
from mloptimizer.domain.hyperspace import HyperparameterSpace, Hyperparam
from mloptimizer.domain.evaluation import kfold_stratified_score
from mloptimizer.application.reporting.plots import plotly_search_space

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.datasets import load_iris

from xgboost import XGBClassifier

from hyperopt import STATUS_OK, hp, tpe
from hyperopt import Trials, fmin

width = 1000
height = 1000

# %%
# 1) Dataset Description
# ----------------------
# The Iris dataset is a classic dataset used for classification tasks.
# It consists of 150 samples, each representing an iris flower.
#
# Features:
#
# - Sepal Length: Length of the sepal in centimeters.
# - Sepal Width: Width of the sepal in centimeters.
# - Petal Length: Length of the petal in centimeters.
# - Petal Width: Width of the petal in centimeters.
#
# Target:
#
# - Species of the flower, which can be one of the following three classes:
#
#   1. Setosa
#   2. Versicolor
#   3. Virginica
#
# Characteristics:
#
# - The dataset is balanced, containing 50 samples for each class.
# - Features are continuous and have varying scales, which may require normalization or standardization for certain machine learning algorithms.

name = 'iris'
X, y = load_iris(return_X_y=True)

print(f"1) Description of the dataset")
print(f"Dataset: {name}, X shape: {X.shape}, y shape: {y.shape}")

# %%
# 2) Genetic Optimization of XGBoost Algorithm
# --------------------------------------------
# Genetic optimization is performed using the mloptimizer library to fine-tune the hyperparameters of the XGBoost
# algorithm.
#
# Hyperparameters to Optimize:
#
# - `colsample_bytree`: Subsample ratio of columns when constructing each tree.
# - `gamma`: Minimum loss reduction required to make a further partition on a leaf node of the tree.
# - `learning_rate`: Step size shrinkage used in updates to prevent overfitting.
# - `max_depth`: Maximum depth of a tree.
# - `n_estimators`: Number of boosting rounds.
# - `subsample`: Subsample ratio of the training instances.
#
# Optimization Process:
#
# - Population Size: 10
# - Generations: 10
# - Fitness Score: Balanced accuracy
# - Evaluation Function: Stratified k-fold cross-validation with 5 folds
#
# The genetic optimization explores the hyperparameter space defined by the evolvable hyperparameters.
# It searches within the defined minimum and maximum values for each hyperparameter.
#
# Advantages:
#
# - Provides a more exhaustive search compared to grid search and random search.
# - Can be more efficient in finding optimal hyperparameters for the XGBoost algorithm.

print(f"2) Genetic optimization of the algorithm XGBoost")

fixed_hyperparams = {}
evolvable_hyperparams = {
    'colsample_bytree': Hyperparam('colsample_bytree', 3, 10, 'float', 10),
    'gamma': Hyperparam('gamma', 0, 20, 'int'),
    'learning_rate': Hyperparam('learning_rate', 1, 100, 'float', 1000),
    'max_depth': Hyperparam('max_depth', 2, 20, 'int'),
    'n_estimators': Hyperparam('n_estimators', 100, 500, 'int'),
    'subsample': Hyperparam('subsample', 700, 1000, 'float', 1000)
}
hyperparameter_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)

opt = Optimizer(
    estimator_class=XGBClassifier,
    features=X,
    labels=y,
    hyperparam_space=hyperparameter_space,
    eval_function=kfold_stratified_score,
    fitness_score="balanced_accuracy", seed=0,
    use_parallel=False
)
population_size = 10
generations = 10
t0_gen = time()
clf = opt.optimize_clf(population_size=population_size, generations=generations)  # Aprox 100 elements
t1_gen = time()
print(f"Genetic optimization around {population_size * (generations + 1)} algorithm executions")
execution_time_gen = round(t1_gen - t0_gen, 2)
print(f"Time of the genetic optimization {execution_time_gen} s")
population_df = opt.runs[0].population_2_df()
print(f"Genetic optimization {population_df.shape[0]} algorithm executions")
df = population_df[list(hyperparameter_space.evolvable_hyperparams.keys()) + ['fitness']]
fig_gen = plotly_search_space(df).update_layout(
    autosize=True,
    width=width,  # Adjust width as needed
    height=height,  # Adjust height as needed
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins as needed
)
plotly.io.show(fig_gen)

# %%
# 3) Grid Search Optimization for XGBoost
# ---------------------------------------
# Grid Search optimization is performed using the GridSearchCV class from the scikit-learn library.
#
# Optimization Process:
#
# - The hyperparameters to optimize are the same as those used in genetic optimization:
#
#   - `colsample_bytree`
#   - `gamma`
#   - `learning_rate`
#   - `max_depth`
#   - `n_estimators`
#   - `subsample`
#
# - Search Strategy:
#
#   - Grid Search performs an exhaustive search over a predefined hyperparameter space.
#   - The search space and number of parameter combinations are fixed by the user.
#
# Considerations:
#
# - Grid Search is computationally expensive, especially for large hyperparameter spaces.
# - It may not be feasible for extensive parameter tuning due to its exhaustive nature.

print(f"3) Grid Search of the algorithm Grid Search")


xgb = XGBClassifier()
parameters = {
    'colsample_bytree': (0.3, 0.5, 0.8),
    'gamma': (5, 20),
    'learning_rate': (0.001, 0.01, 0.1),
    'max_depth': (2, 10),
    'n_estimators': (300,),
    'subsample': (0.7, 0.8, 0.9)
}
gs_executions = reduce(lambda x, y: x * y, [len(parameters[k]) for k in parameters.keys()], 1)
print(f"Grid Search optimization will run {gs_executions} algorithm executions")
clf_gs = GridSearchCV(
    xgb,
    parameters,
    cv=5,
    scoring="balanced_accuracy",
)

t0_gs = time()
clf_gs.fit(X, y)
t1_gs = time()
execution_time_gs = round(t1_gs - t0_gs, 2)

print(f"Time of the grid search {execution_time_gs} s")

synth_population_gs = pd.DataFrame(clf_gs.cv_results_['params'])
synth_population_gs['fitness'] = clf_gs.cv_results_['mean_test_score']
fig_gs = plotly_search_space(synth_population_gs).update_layout(
    autosize=True,
    width=width,  # Adjust width as needed
    height=height,  # Adjust height as needed
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins as needed
)
plotly.io.show(fig_gs)

print(f"Grid Search optimization has run {synth_population_gs.shape[0]} algorithm executions")

# %%
# 4) Random Search Optimization for XGBoost
# -----------------------------------------
# Random Search optimization is performed using the RandomizedSearchCV class from the scikit-learn library.
#
# Optimization Process:
#
# - The hyperparameters to optimize are the same as those used in genetic optimization and grid search:
#
#   - `colsample_bytree`
#   - `gamma`
#   - `learning_rate`
#   - `max_depth`
#   - `n_estimators`
#   - `subsample`
#
# - Search Strategy:
#
#   - Random Search samples a specified number of random combinations from the predefined hyperparameter space.
#   - The number of iterations (i.e., sampled parameter combinations) is fixed by the user.
#
# Considerations:
#
# - Random Search is less computationally expensive than Grid Search, making it more feasible for larger search spaces.
# - It is not as exhaustive as genetic optimization but can be more efficient in exploring the hyperparameter space compared to Grid Search.

print(f"4) Random Search of the algorithm")
distributions = {
    'colsample_bytree': np.linspace(0.3, 1, 10),
    'gamma': (0, 5, 20),
    'learning_rate': (0.001, 0.01, 0.1),
    'max_depth': (2, 5, 10, 20),
    'n_estimators': (100, 300, 500),
    'subsample': np.linspace(0.7, 0.9, 10)
}
rs_executions = reduce(lambda x, y: x * y, [len(distributions[k]) for k in distributions.keys()], 1)
print(f"Random Search optimization could run {rs_executions} different algorithm executions")
clf_rs = RandomizedSearchCV(
    xgb,
    distributions,
    cv=5,
    n_iter=110,
    random_state=0,
    scoring="balanced_accuracy"
)

t0_rs = time()
search = clf_rs.fit(X, y)
t1_rs = time()
execution_time_rs = round(t1_rs - t0_rs, 2)
print(f"Time of the grid search {execution_time_rs} s")

synth_population_rs = pd.DataFrame(clf_rs.cv_results_['params'])
synth_population_rs['fitness'] = clf_rs.cv_results_['mean_test_score']
fig_rs = plotly_search_space(synth_population_rs).update_layout(
    autosize=True,
    width=width,  # Adjust width as needed
    height=height,  # Adjust height as needed
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins as needed
)
print(f"Random Search optimization has run {synth_population_rs.shape[0]} algorithm executions")
plotly.io.show(fig_rs)

# %% 5) Bayesian Optimization for XGBoost
# ---------------------------------------
# Bayesian Optimization is performed using the hyperopt library to optimize the hyperparameters of the XGBoost algorithm.
#
# Optimization Process:
#
# - The hyperparameters to optimize are the same as those used in genetic optimization, grid search, and random search:
#
#   - `colsample_bytree`
#   - `gamma`
#   - `learning_rate`
#   - `max_depth`
#   - `n_estimators`
#   - `subsample`
#
# - Search Strategy:
#
#   - Bayesian Optimization uses a probabilistic model to predict the performance of different hyperparameter combinations.
#   - It iteratively evaluates the objective function (in this case, cross-validated balanced accuracy) to find the optimal hyperparameters.
#
# Considerations:
#
# - Bayesian Optimization is more efficient than random search and grid search for large search spaces.
# - It can provide better results with fewer evaluations compared to random search and grid search.
# - The number of evaluations is a parameter that can be adjusted to balance the trade-off between performance and computational cost.

print(f"5) Bayesian Optimization of the algorithm")
bayesian_space = {
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.3, 1.0),
    'gamma': hp.choice('gamma', np.arange(5, 50+1, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(2, 10+1, dtype=int)),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 500, dtype=int)),
    'subsample': hp.uniform('subsample', 0.6, 1.0)
}

def objective_function(params):
    clf = XGBClassifier(**params)
    score = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean()
    return {'loss': -score, 'status': STATUS_OK}

trials = Trials()
num_eval = 110

print(f"Bayesian Search optimization will run {num_eval} algorithm executions")

t0_bay = time()
best_param = fmin(objective_function, bayesian_space, algo=tpe.suggest, max_evals=num_eval, trials=trials)
t1_bay = time()
execution_time_bay = round(t1_bay - t0_bay, 2)
print(f"Time of the bayesian search {execution_time_bay} s")


# Extract parameters and scores from trials
trials_dict = trials.trials
params_list = [{k: v[0] for k, v in trial['misc']['vals'].items()} for trial in trials_dict]
scores_list = [-trial['result']['loss'] for trial in trials_dict]
# Convert parameters to DataFrame
synth_population_bay = pd.DataFrame(params_list)
# Add scores to DataFrame
synth_population_bay['fitness'] = scores_list
# Rename columns to match the parameter names
synth_population_bay.columns = [col.replace('vals_', '') for col in synth_population_bay.columns]

fig_bay = plotly_search_space(synth_population_bay).update_layout(
    autosize=True,
    width=width,  # Adjust width as needed
    height=height,  # Adjust height as needed
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins as needed
)
print(f"Bayesian Search optimization has run {num_eval} algorithm executions")
plotly.io.show(fig_bay)



# %%
# Summary Table
# -------------
# The summary table below compares the optimization methods based on their best metric,
# the time taken for optimization, and the number of evaluations performed:
#
#
# Overview of Optimization Methods:
#
# - **Genetic Optimization**:
#
#   - Achieved the highest metric (0.961004) with 110 evaluations.
#   - Completed in the shortest time (25.87 seconds).
#   - Utilizes evolutionary algorithms to perform the search in the hyperparameter space.
#   - Efficient in finding optimal hyperparameters.
#
# - **Grid Search**:
#
#   - Achieved a best metric of 0.960000 with 108 evaluations.
#   - Took the longest time to complete (69.11 seconds).
#   - Performs an exhaustive search over a predefined grid of hyperparameters.
#   - Computationally expensive, particularly for large search spaces.
#
# - **Random Search**:
#
#   - Achieved a best metric of 0.960000 with 110 evaluations.
#   - Took less time than Grid Search (58.89 seconds).
#   - Samples random combinations of hyperparameters from a predefined space.
#   - Less computationally expensive than Grid Search and more efficient for large spaces.
#
# Conclusions:
#
# - Genetic Optimization provided the best performance in terms of both metric and time.
# - Grid Search is the most computationally expensive method.
# - Random Search is a good compromise between computational cost and performance.

row_gen = {'Method': "Genetic",
           'Best Metric': float(population_df.sort_values(by="fitness", ascending=False)['fitness'].iloc[0]),
           'Time': execution_time_gen, 'Evaluated': population_df.shape[0]}
row_gs = {'Method': "Grid Search", 'Best Metric': clf_gs.best_score_,
          'Time': execution_time_gs, 'Evaluated': synth_population_gs.shape[0]}
row_rs = {'Method': "Random Search", 'Best Metric': clf_rs.best_score_,
          'Time': execution_time_rs, 'Evaluated': synth_population_rs.shape[0]}
row_bay = {'Method': "Bayesian Search",
           'Best Metric': max(scores_list),
           'Time': execution_time_bay, 'Evaluated': num_eval}

df = pd.DataFrame([row_gen, row_gs, row_rs, row_bay])
df
