"""
Regression Example with RandomForest
=====================================
Hyperparameter optimization for RandomForestRegressor using genetic algorithms.
This example demonstrates regression optimization with visualization of the search space
and evolution of fitness scores.
"""

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch
from mloptimizer.application.reporting.plots import plotly_search_space, plotly_logbook

# %%
# Load and prepare the dataset
# ----------------------------
# The diabetes dataset is a regression dataset with 10 features predicting disease progression.
print("Loading Diabetes dataset...")
data = load_diabetes()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")

# %%
# Visualize the relationship between BMI and disease progression
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 2], y, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel("Body Mass Index (normalized)")
plt.ylabel("Disease Progression")
plt.title("Diabetes Dataset: BMI vs Disease Progression")
# sphinx-gallery captures the plot automatically

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %%
# Define the hyperparameter space
# -------------------------------
# Using a custom space with reduced n_estimators for faster documentation builds.
# For production, use the default space or increase the ranges.
hyperparam_space = (HyperparameterSpaceBuilder()
                    .add_int_param('n_estimators', min_value=20, max_value=100)
                    .add_int_param('max_depth', min_value=3, max_value=15)
                    .add_int_param('min_samples_split', min_value=2, max_value=15)
                    .add_int_param('min_samples_leaf', min_value=1, max_value=10)
                    .add_float_param('max_features', min_value=30, max_value=100, scale=100)
                    .build())

# For production, use the default space instead:
# hyperparam_space = HyperparameterSpaceBuilder.get_default_space(RandomForestRegressor)

print(f"Evolvable parameters: {list(hyperparam_space.evolvable_hyperparams.keys())}")

# %%
# Configure and run the genetic optimization
# ------------------------------------------
# Note: Values reduced for faster documentation builds.
genetic_params = {
    'generations': 3,
    'population_size': 6,
    'n_elites': 2,
    'seed': 42,
    'use_parallel': False
}

opt = GeneticSearch(
    estimator_class=RandomForestRegressor,
    hyperparam_space=hyperparam_space,
    cv=3,
    scoring='rmse',  # Root Mean Squared Error (also available: 'mse')
    **genetic_params
)

print("Starting RandomForestRegressor optimization...")
opt.fit(X_train, y_train)

# %%
# Evaluate the optimized model
best_reg = opt.best_estimator_
y_pred = best_reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print(f"\nOptimization completed!")
print(f"Best parameters: {opt.best_params_}")
print(f"\nTest performance:")
print(f"  MSE:  {test_mse:.2f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  R2:   {test_r2:.4f}")

# %%
# Visualize predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"RandomForest Regression: Predictions vs Actual (R² = {test_r2:.3f})")
# sphinx-gallery captures the plot automatically

# %%
# Visualize the search space
population_df = opt.populations_
top_params = ['n_estimators', 'max_depth', 'min_samples_split', 'max_features', 'fitness']
df_filtered = population_df[top_params]
g_search_space = plotly_search_space(df_filtered, top_params)
g_search_space.update_layout(
    title="RandomForestRegressor Hyperparameter Search Space",
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
    title="RandomForestRegressor Optimization Evolution",
    autosize=True,
    width=None,
    height=500
)
# sphinx-gallery captures the figure automatically
g_logbook

# %%
# Analyze optimization results
print("\n=== Optimization Analysis ===")
print(f"Unique evaluations performed: {opt.n_trials_}")
print(f"Total individuals in population history: {len(population_df)}")
print(f"Optimization time: {opt.optimization_time_:.4f} seconds")
print(f"Time per evaluation: {opt.optimization_time_ / opt.n_trials_:.4f} seconds")
print(f"Generations completed: {opt.generations}")

final_gen = population_df[population_df['population'] == opt.generations]
initial_gen = population_df[population_df['population'] == 1]

final_avg_fitness = final_gen['fitness'].mean()
initial_avg_fitness = initial_gen['fitness'].mean()
improvement = final_avg_fitness - initial_avg_fitness

print(f"\nFitness Evolution:")
print(f"  Initial average fitness: {initial_avg_fitness:.4f}")
print(f"  Final average fitness:   {final_avg_fitness:.4f}")
print(f"  Average improvement:     {improvement:.4f}")

# %%
# Available regression scoring metrics
# ------------------------------------
# mloptimizer supports the following regression metrics:
#
# - ``rmse``: Root Mean Squared Error (default for regressors)
# - ``mse``: Mean Squared Error
#
# For regressors, the fitness is minimized (lower is better).
#
# Example with MSE scoring:
#
# .. code-block:: python
#
#     opt = GeneticSearch(
#         estimator_class=RandomForestRegressor,
#         hyperparam_space=hyperparam_space,
#         cv=5,
#         scoring='mse',  # Optimize for MSE
#         generations=10,
#         population_size=20
#     )
