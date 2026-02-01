"""
Population Seeding with initial_params
======================================

This example demonstrates how to seed the initial population with known good
hyperparameter configurations using the ``initial_params`` feature.

Population seeding is useful when you:

- Have prior knowledge of good hyperparameter ranges from previous experiments
- Want to warm-start the genetic algorithm with configurations from GridSearch
- Need to refine hyperparameters around a known good starting point
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch

# %%
# Load and split the dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Define the hyperparameter space
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
    estimator_class=DecisionTreeClassifier
)

# %%
# Define known good hyperparameter configurations to seed the population.
# These could come from previous experiments, domain knowledge, or GridSearch results.
initial_configs = [
    {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 2},
    {'max_depth': 3, 'min_samples_split': 5, 'min_samples_leaf': 1},
    {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 4},
]

# %%
# Run optimization WITH population seeding
# The genetic algorithm will start with these configurations plus random ones
opt_seeded = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    initial_params=initial_configs,
    include_default=True,  # Also include sklearn defaults
    generations=5,
    population_size=10,
    seed=42,
    use_parallel=False  # For reproducibility in this example
)

opt_seeded.fit(X_train, y_train)

print("=== With Population Seeding ===")
print(f"Best score: {opt_seeded.best_score_:.4f}")
print(f"Best params: {opt_seeded.best_params_}")

# %%
# Run optimization WITHOUT population seeding for comparison
opt_random = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    initial_params=None,  # Random initialization
    generations=5,
    population_size=10,
    seed=123,  # Different seed for fair comparison
    use_parallel=False
)

opt_random.fit(X_train, y_train)

print("\n=== Without Population Seeding ===")
print(f"Best score: {opt_random.best_score_:.4f}")
print(f"Best params: {opt_random.best_params_}")

# %%
# Evaluate on test set
test_score_seeded = opt_seeded.best_estimator_.score(X_test, y_test)
test_score_random = opt_random.best_estimator_.score(X_test, y_test)

print("\n=== Test Set Evaluation ===")
print(f"Seeded optimization test score: {test_score_seeded:.4f}")
print(f"Random optimization test score: {test_score_random:.4f}")

# %%
# Using include_default parameter
# --------------------------------
# The ``include_default`` parameter adds sklearn's default hyperparameters
# to the initial population, which can be a good baseline.

opt_with_defaults = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    initial_params=[{'max_depth': 5}],  # Just one custom config
    include_default=True,  # Also include sklearn defaults
    generations=3,
    population_size=8,
    seed=42,
    use_parallel=False
)

opt_with_defaults.fit(X_train, y_train)
print(f"\nWith include_default=True, best score: {opt_with_defaults.best_score_:.4f}")

# %%
# Partial hyperparameter specification
# ------------------------------------
# You don't need to specify all hyperparameters. Unspecified ones
# will be randomly sampled from the search space.

partial_config = [
    {'max_depth': 5},  # Only specify max_depth
]

opt_partial = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    initial_params=partial_config,
    generations=3,
    population_size=8,
    seed=42,
    use_parallel=False
)

opt_partial.fit(X_train, y_train)
print(f"With partial config, best score: {opt_partial.best_score_:.4f}")

# %%
# Best practices for population seeding
# -------------------------------------
#
# 1. **Don't overfill**: Keep ``initial_params`` smaller than ``population_size``
#    to maintain genetic diversity
#
# 2. **Diverse seeds**: Provide configurations that explore different regions
#    of the hyperparameter space
#
# 3. **Combine with include_default**: Adding sklearn defaults ensures you
#    have at least one reasonable baseline
#
# 4. **Use with early stopping**: Population seeding + early stopping can
#    quickly converge to good solutions
