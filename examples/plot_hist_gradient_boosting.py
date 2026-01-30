"""
HistGradientBoosting Optimization
==================================
Hyperparameter optimization for sklearn's fast HistGradientBoosting algorithms.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import plotly
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch
from mloptimizer.application.reporting.plots import plotly_search_space, plotly_logbook

# %%
# Load and prepare the dataset
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# Define the hyperparameter space
# Using custom space with reduced max_iter for faster documentation builds.
# For production, use the default space or increase max_iter range.
hyperparam_space = (HyperparameterSpaceBuilder()
                    .add_float_param('learning_rate', min_value=1, max_value=100, scale=1000)
                    .add_int_param('max_depth', min_value=2, max_value=10)
                    .add_int_param('max_iter', min_value=20, max_value=80)
                    .add_int_param('max_leaf_nodes', min_value=20, max_value=60)
                    .add_int_param('min_samples_leaf', min_value=10, max_value=30)
                    .build())

# For production, use the default space instead:
# hyperparam_space = HyperparameterSpaceBuilder.get_default_space(HistGradientBoostingClassifier)

# %%
# Configure and run the genetic optimization
genetic_params = {
    'generations': 3,
    'population_size': 6,
    'n_elites': 2,
    'seed': 42,
    'use_mlflow': False,
    'use_parallel': False
}

opt = GeneticSearch(
    estimator_class=HistGradientBoostingClassifier,
    hyperparam_space=hyperparam_space,
    cv=3,
    scoring='accuracy',
    **genetic_params
)

print("Starting HistGradientBoostingClassifier optimization...")
opt.fit(X_train, y_train)

# %%
# Evaluate the optimized model
best_clf = opt.best_estimator_
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='binary')

print(f"\nOptimization completed!")
print(f"Best parameters: {opt.best_params_}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test F1: {test_f1:.4f}")

# %%
# Visualize the search space
population_df = opt.populations_
top_params = ['learning_rate', 'max_depth', 'max_iter', 'min_samples_leaf', 'fitness']
df_filtered = population_df[top_params]
g_search_space = plotly_search_space(df_filtered, top_params)
g_search_space.update_layout(
    title="HistGradientBoostingClassifier Hyperparameter Search Space",
    autosize=True,
    width=None,
    height=650
)
# sphinx-gallery captures the figure automatically
g_search_space

# %%
# Visualize the optimization evolution
g_logbook = plotly_logbook(opt.logbook_, population_df)
g_logbook.update_layout(
    title="HistGradientBoostingClassifier Optimization Evolution",
    autosize=True,
    width=None,
    height=500
)
# sphinx-gallery captures the figure automatically
g_logbook

# %%
# Analyze optimization performance
print("\n=== Optimization Performance ===")
print(f"Unique evaluations performed: {opt.n_trials_}")
print(f"Total individuals in population history: {len(population_df)}")
print(f"Optimization time: {opt.optimization_time_:.4f} seconds")
print(f"Time per evaluation: {opt.optimization_time_ / opt.n_trials_:.4f} seconds")
