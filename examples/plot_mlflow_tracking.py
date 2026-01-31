"""
MLflow Experiment Tracking
==========================

This example demonstrates how to use MLflow integration with mloptimizer
to track hyperparameter optimization experiments.

MLflow is an optional dependency. Install it with: ``pip install mlflow``
"""

# %%
# Setup and Imports
# -----------------
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

# %%
# Configure MLflow Backend (Recommended)
# --------------------------------------
# The file-based backend (./mlruns) was deprecated by MLflow in February 2026.
# We recommend configuring a database backend before using MLflow.
#
# Options:
# - SQLite (no extra dependencies): ``sqlite:///mlflow.db``
# - PostgreSQL: ``postgresql://user:password@host:5432/mlflow``
# - MySQL: ``mysql://user:password@host:3306/mlflow``

mlflow.set_tracking_uri("sqlite:///mlflow.db")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# %%
# Load and Prepare Data
# ---------------------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %%
# Define Hyperparameter Space
# ---------------------------
space = (HyperparameterSpaceBuilder()
         .add_int_param("n_estimators", 50, 200)
         .add_int_param("max_depth", 3, 15)
         .add_int_param("min_samples_split", 2, 20)
         .add_int_param("min_samples_leaf", 1, 10)
         .build())

print(f"Hyperparameter space: {space.evolvable_hyperparams.keys()}")

# %%
# Run Optimization with MLflow Tracking
# -------------------------------------
# When ``use_mlflow=True``, mloptimizer logs:
#
# **Parent Run (optimization-level):**
#   - GA parameters (population_size, generations, etc.)
#   - Generation metrics (best/avg/worst fitness per generation)
#   - Dataset metadata (samples, features, classes)
#   - Final best fitness
#
# **Child Runs (individual evaluations):**
#   - Hyperparameters for each individual
#   - Fitness score
#   - Generation and individual index
#
# MLflow logging works seamlessly with both parallel and sequential execution.
# When using parallel mode, child runs are logged in batches after each
# generation completes.

opt = GeneticSearch(
    estimator_class=RandomForestClassifier,
    hyperparam_space=space,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="balanced_accuracy",
    generations=5,
    population_size=10,
    n_elites=2,
    seed=42,
    use_mlflow=True,       # Enable MLflow tracking
    use_parallel=True,     # Works with parallel execution
)

print("\nStarting optimization with MLflow tracking...")
opt.fit(X_train, y_train)

# %%
# Evaluate Results
# ----------------
best_clf = opt.best_estimator_
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nOptimization completed!")
print(f"Best parameters: {opt.best_params_}")
print(f"Total evaluations: {opt.n_trials_}")
print(f"Test accuracy: {test_accuracy:.4f}")

# %%
# Query MLflow Data Programmatically
# ----------------------------------
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("mloptimizer")

if experiment:
    runs = client.search_runs([experiment.experiment_id], max_results=5)
    print(f"\nMLflow Experiment: {experiment.name}")
    print(f"Total runs: {len(runs)}")

    # Show recent parent runs
    parent_runs = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    if parent_runs:
        print(f"\nRecent optimization runs:")
        for run in parent_runs[:3]:
            print(f"  - {run.info.run_name}: fitness={run.data.metrics.get('final_best_fitness', 'N/A')}")

# %%
# View Results in MLflow UI
# -------------------------
# Start the MLflow UI to visualize your experiments:
#
# .. code-block:: bash
#
#    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
#
# Then open http://localhost:5000 in your browser.
#
# **What to explore in the UI:**
#
# 1. **Experiments view**: See all optimization runs
# 2. **Run details**: Click a run to see parameters and metrics
# 3. **Metrics charts**: Click ``generation_best_fitness`` to see evolution curve
# 4. **Compare runs**: Select multiple runs to compare hyperparameters
# 5. **Child runs**: Expand a parent run to see individual evaluations

print("\n" + "="*60)
print("To view results in MLflow UI, run:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
print("Then open: http://localhost:5000")
print("="*60)
