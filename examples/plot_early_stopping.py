"""
Early Stopping
==============

This example demonstrates how to use early stopping to automatically terminate
the optimization when no significant improvement is observed.

Early stopping is useful when you:

- Want to avoid wasting computation on stagnant optimization
- Need to run many optimizations and want automatic termination
- Are exploring large hyperparameter spaces where convergence time varies
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch

# %%
# Load and split the dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Basic early stopping
# --------------------
# Enable early stopping with the ``early_stopping`` parameter.
# The optimization will stop if no improvement is seen for ``patience`` generations.

hyperparam_space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

opt_early = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    generations=50,  # Maximum generations (may stop earlier)
    population_size=10,
    early_stopping=True,
    patience=5,  # Stop if no improvement for 5 generations
    seed=42,
    use_parallel=False,
    verbose=1  # Show optimization progress
)

opt_early.fit(X_train, y_train)

print(f"Best score: {opt_early.best_score_:.4f}")
print(f"Total trials: {opt_early.n_trials_}")
print(f"Optimization time: {opt_early.optimization_time_:.2f}s")

# %%
# Compare with and without early stopping
# ---------------------------------------

# Without early stopping - runs all 20 generations
opt_full = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    generations=20,
    population_size=10,
    early_stopping=False,
    seed=42,
    use_parallel=False
)
opt_full.fit(X_train, y_train)

# With early stopping - may stop before 20 generations
opt_early2 = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    generations=20,
    population_size=10,
    early_stopping=True,
    patience=3,
    seed=42,
    use_parallel=False
)
opt_early2.fit(X_train, y_train)

print("\n=== Comparison ===")
print(f"Without early stopping: {opt_full.n_trials_} trials, score={opt_full.best_score_:.4f}")
print(f"With early stopping: {opt_early2.n_trials_} trials, score={opt_early2.best_score_:.4f}")

# %%
# Using min_delta for improvement threshold
# -----------------------------------------
# The ``min_delta`` parameter specifies the minimum improvement required
# to reset the patience counter.

opt_sensitive = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    generations=30,
    population_size=10,
    early_stopping=True,
    patience=5,
    min_delta=0.01,  # Require at least 1% improvement
    seed=42,
    use_parallel=False
)
opt_sensitive.fit(X_train, y_train)

opt_lenient = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    generations=30,
    population_size=10,
    early_stopping=True,
    patience=5,
    min_delta=0.001,  # Only 0.1% improvement needed
    seed=42,
    use_parallel=False
)
opt_lenient.fit(X_train, y_train)

print("\n=== min_delta Comparison ===")
print(f"min_delta=0.01: {opt_sensitive.n_trials_} trials")
print(f"min_delta=0.001: {opt_lenient.n_trials_} trials")

# %%
# Early stopping with RandomForest
# --------------------------------
# Early stopping is especially useful with slower models like RandomForest.

rf_space = HyperparameterSpaceBuilder.get_default_space(RandomForestClassifier)

opt_rf = GeneticSearch(
    estimator_class=RandomForestClassifier,
    hyperparam_space=rf_space,
    generations=20,
    population_size=8,
    early_stopping=True,
    patience=3,
    min_delta=0.005,
    seed=42,
    use_parallel=False
)
opt_rf.fit(X_train, y_train)

print(f"\nRandomForest optimization:")
print(f"Best score: {opt_rf.best_score_:.4f}")
print(f"Trials: {opt_rf.n_trials_}")
print(f"Time: {opt_rf.optimization_time_:.2f}s")

# %%
# Test set evaluation
test_score = opt_rf.best_estimator_.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")

# %%
# Best practices for early stopping
# ---------------------------------
#
# 1. **Start with patience=5-10**: Lower patience stops earlier but may miss improvements
#
# 2. **Use min_delta for noisy metrics**: Set min_delta > 0 if your fitness function is noisy
#
# 3. **Combine with population seeding**: Seeding + early stopping can quickly converge
#
# 4. **Set reasonable max generations**: Even with early stopping, set a sensible maximum
#
# 5. **Monitor with verbose=1**: Use verbose mode to see when early stopping triggers
