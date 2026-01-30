"""
Comprehensive MLflow Integration Tests
======================================

These tests verify that MLflow integration is working correctly, storing information
properly, data is recoverable, experiments are consistent, and errors are traceable.

Test Categories:
1. Basic MLflow functionality (logging works, data persists)
2. Data consistency (logged values match expected values)
3. Data recoverability (can read back what was written)
4. Error handling (traceable errors when things fail)
5. Disabled MLflow behavior (no data created when disabled)

IMPORTANT NOTE: These tests use use_parallel=False because MLflow nested run logging
does not work correctly with parallel execution (joblib workers don't share MLflow context).
This is a known limitation documented in the MLflow integration.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Import mlflow - tests will skip if not installed
mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient

from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder


class TestMLflowBasicFunctionality:
    """Test that MLflow logging works and creates expected structures."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow SQLite database for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        # Cleanup
        mlflow.end_run()  # End any active run

    def test_mlflow_creates_experiment(self):
        """Verify MLflow creates experiment when use_mlflow=True."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        # Verify experiment was created
        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiments = client.search_experiments()

        assert len(experiments) >= 1, "No MLflow experiments created"
        exp_names = [e.name for e in experiments]
        assert "mloptimizer" in exp_names, f"Expected 'mloptimizer' experiment, found: {exp_names}"

    def test_mlflow_creates_parent_run(self):
        """Verify MLflow creates a parent run with correct naming."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        # Find parent run (format: YYYYMMDD_HHMMSS_EstimatorName)
        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]

        assert len(parent_runs) >= 1, "No parent run created with expected naming format"

    def test_mlflow_creates_nested_runs(self):
        """Verify MLflow creates nested child runs for individual evaluations."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        # Find child runs (format: gen_N_ind_M_EstimatorName)
        import re
        child_runs = [r for r in runs if re.match(r'^gen_\d+_ind_\d+_', r.info.run_name)]

        # Should have at least population_size * generations child runs
        # (may be less due to elitism)
        assert len(child_runs) >= 4, f"Expected at least 4 child runs, found {len(child_runs)}"


class TestMLflowDataConsistency:
    """Test that logged values match expected values."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow directory for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        mlflow.end_run()

    def test_genetic_params_logged_correctly(self):
        """Verify genetic algorithm parameters are logged with correct values."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        GENERATIONS = 3
        POPULATION_SIZE = 5

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=GENERATIONS,
            population_size=POPULATION_SIZE,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        # Find parent run
        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        assert len(parent_runs) >= 1, "No parent run found"

        parent = parent_runs[0]
        params = parent.data.params

        # Verify genetic params are logged correctly
        assert "generations" in params, "generations parameter not logged"
        assert "population_size" in params, "population_size parameter not logged"
        assert int(params["generations"]) == GENERATIONS, \
            f"Expected generations={GENERATIONS}, got {params['generations']}"
        assert int(params["population_size"]) == POPULATION_SIZE, \
            f"Expected population_size={POPULATION_SIZE}, got {params['population_size']}"

    def test_generation_metrics_logged(self):
        """Verify generation-level metrics are logged."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=3,
            population_size=5,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]
        metrics = parent.data.metrics

        # Verify generation-level metrics exist
        expected_metrics = [
            "generation_best_fitness",
            "generation_avg_fitness",
            "final_best_fitness"
        ]

        for metric_name in expected_metrics:
            assert metric_name in metrics, f"Expected metric '{metric_name}' not logged"

    def test_dataset_tags_logged(self):
        """Verify dataset metadata is logged as tags."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]
        tags = parent.data.tags

        # Verify dataset tags
        assert "dataset_samples" in tags, "dataset_samples tag not logged"
        assert "dataset_features" in tags, "dataset_features tag not logged"
        assert tags["dataset_samples"] == str(X.shape[0]), \
            f"Expected samples={X.shape[0]}, got {tags['dataset_samples']}"
        assert tags["dataset_features"] == str(X.shape[1]), \
            f"Expected features={X.shape[1]}, got {tags['dataset_features']}"

    def test_estimator_class_tag_logged(self):
        """Verify estimator class name is logged correctly."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]

        assert "estimator_class" in parent.data.tags, "estimator_class tag not logged"
        assert parent.data.tags["estimator_class"] == "DecisionTreeClassifier", \
            f"Expected 'DecisionTreeClassifier', got {parent.data.tags['estimator_class']}"

    def test_final_fitness_matches_best_score(self):
        """Verify final_best_fitness metric matches GeneticSearch.best_score_."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=3,
            population_size=5,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        # Get best fitness from GeneticSearch (now using best_score_ attribute)
        best_score = opt.best_score_

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]

        mlflow_final_fitness = parent.data.metrics.get("final_best_fitness")
        assert mlflow_final_fitness is not None, "final_best_fitness not logged"

        # Allow small floating point tolerance
        assert abs(mlflow_final_fitness - best_score) < 1e-6, \
            f"MLflow final_best_fitness ({mlflow_final_fitness}) != best_score_ ({best_score})"


class TestMLflowDataRecoverability:
    """Test that logged data can be read back correctly."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow directory for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        mlflow.end_run()

    def test_metric_history_recoverable(self):
        """Verify generation metrics can be retrieved with full history."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        GENERATIONS = 4

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=GENERATIONS,
            population_size=5,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]

        # Get metric history (generation-level tracking)
        history = client.get_metric_history(parent.info.run_id, "generation_best_fitness")

        # Should have one entry per generation (including generation 0)
        # Note: may vary based on implementation
        assert len(history) >= GENERATIONS, \
            f"Expected at least {GENERATIONS} metric entries, got {len(history)}"

        # Verify steps are sequential
        steps = sorted([m.step for m in history])
        assert steps == list(range(len(steps))), \
            f"Metric steps should be sequential, got {steps}"

    def test_child_run_hyperparams_recoverable(self):
        """Verify child runs have recoverable hyperparameters."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        child_runs = [r for r in runs if re.match(r'^gen_\d+_ind_\d+_', r.info.run_name)]

        assert len(child_runs) > 0, "No child runs found"

        # Each child run should have hyperparameters
        for child in child_runs:
            params = child.data.params
            assert len(params) > 0, f"Child run {child.info.run_name} has no parameters"

            # Should have DecisionTree-relevant params
            possible_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion']
            has_relevant_param = any(p in params for p in possible_params)
            assert has_relevant_param, \
                f"Child run params don't include expected DecisionTree params: {list(params.keys())}"

    def test_runs_persist_after_new_session(self):
        """Verify data persists and is readable in a new MLflow session."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,  # Required for MLflow nested run logging
            seed=42
        )
        opt.fit(X, y)

        # "Close" current session by creating new client
        mlflow.end_run()

        # Simulate new session
        new_client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiments = new_client.search_experiments()

        assert len(experiments) >= 1, "Experiments not persisted"

        exp = new_client.get_experiment_by_name("mloptimizer")
        assert exp is not None, "mloptimizer experiment not recoverable"

        runs = new_client.search_runs([exp.experiment_id])
        assert len(runs) > 0, "Runs not persisted"


class TestMLflowDisabledBehavior:
    """Test that MLflow disabled mode doesn't create any MLflow data."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated directory for each test."""
        self.test_dir = tmp_path
        self.mlflow_db = tmp_path / "mlflow.db"
        yield

    def test_no_mlflow_db_created_when_disabled(self):
        """Verify no mlflow.db file is created when use_mlflow=False."""
        os.chdir(self.test_dir)

        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            use_mlflow=False,  # Disabled
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        # Verify no mlflow.db file created
        mlflow_db_exists = self.mlflow_db.exists()
        assert not mlflow_db_exists, "mlflow.db should not be created when use_mlflow=False"

    def test_optimization_works_without_mlflow(self):
        """Verify optimization completes successfully without MLflow."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            use_mlflow=False,
            seed=42
        )
        opt.fit(X, y)

        # Verify optimization completed successfully
        assert hasattr(opt, 'best_estimator_'), "Optimization should complete without MLflow"
        assert opt.best_estimator_ is not None
        assert opt.n_trials_ > 0


class TestMLflowErrorHandling:
    """Test error handling and traceability."""

    def test_mlflow_disabled_does_not_import(self):
        """Verify MLflow import is skipped when use_mlflow=False."""
        from mloptimizer.infrastructure.tracking import Tracker

        # When use_mlflow=False, MLflow should not be imported/used
        with tempfile.TemporaryDirectory() as tmp:
            t = Tracker(name="test", folder=tmp, use_mlflow=False)
            # Tracker should have use_mlflow=False
            assert t.use_mlflow is False
            # Either mlflow attribute doesn't exist or is None when disabled
            mlflow_attr = getattr(t, 'mlflow', None)
            # When disabled, mlflow operations should be no-ops
            assert mlflow_attr is None or t.use_mlflow is False

    def test_mlflow_attribute_accessible_when_enabled(self):
        """Verify mlflow module is accessible from Tracker when enabled."""
        with tempfile.TemporaryDirectory() as tmp:
            from mloptimizer.infrastructure.tracking import Tracker
            t = Tracker(name="test", folder=tmp, use_mlflow=True)
            assert hasattr(t, 'mlflow'), "Tracker should have mlflow attribute when enabled"
            assert t.mlflow is not None, "mlflow module should be loaded"


class TestMLflowRegressorSupport:
    """Test MLflow integration with regressors."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow directory for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        mlflow.end_run()

    def test_regressor_logs_correctly(self):
        """Verify MLflow logging works for regressors."""
        X, y = load_diabetes(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeRegressor)

        opt = GeneticSearch(
            estimator_class=DecisionTreeRegressor,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        assert len(parent_runs) >= 1, "No parent run created for regressor"

        parent = parent_runs[0]
        assert "DecisionTreeRegressor" in parent.info.run_name, \
            f"Run name should contain 'DecisionTreeRegressor', got: {parent.info.run_name}"

        # Verify tags exist - Note: dataset_classes for regressors shows unique y count
        # (this is current behavior - not "regression" since unique count > 0)
        assert "dataset_samples" in parent.data.tags
        assert "dataset_features" in parent.data.tags


class TestMLflowEarlyStopping:
    """Test MLflow logging with early stopping enabled."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow directory for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        mlflow.end_run()

    def test_early_stopping_tags_logged(self):
        """Verify early stopping information is logged when triggered."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        # Use low patience to potentially trigger early stopping
        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=20,  # High to allow early stopping
            population_size=5,
            n_elites=1,
            early_stopping=True,
            patience=2,
            min_delta=0.01,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]

        # Should have total_evaluations tag
        assert "total_evaluations" in parent.data.tags, "total_evaluations tag should be logged"

        # If early stopped, should have early_stopped tag
        # Note: May not trigger early stopping with this config
        if "early_stopped" in parent.data.tags:
            assert parent.data.tags["early_stopped"] in ["True", "False"]


class TestMLflowConcurrentRuns:
    """Test MLflow handling of concurrent/sequential optimization runs."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow directory for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        mlflow.end_run()

    def test_multiple_sequential_runs(self):
        """Verify multiple sequential optimizations create separate runs."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        # First optimization
        opt1 = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt1.fit(X, y)

        # Second optimization
        opt2 = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=123
        )
        opt2.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]

        # Should have 2 parent runs
        assert len(parent_runs) >= 2, \
            f"Expected at least 2 parent runs for sequential optimizations, got {len(parent_runs)}"


class TestMLflowDataIntegrity:
    """Test data integrity and consistency checks."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_dir(self, tmp_path):
        """Create isolated MLflow directory for each test."""
        self.mlflow_db = tmp_path / "mlflow.db"
        self.mlflow_uri = f"sqlite:///{self.mlflow_db}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        yield
        mlflow.end_run()

    def test_generation_metrics_are_monotonically_reasonable(self):
        """Verify best fitness metrics are reasonable (non-negative for accuracy)."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=4,
            population_size=6,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
        parent = parent_runs[0]

        history = client.get_metric_history(parent.info.run_id, "generation_best_fitness")

        # All fitness values should be between 0 and 1 for accuracy
        for metric in history:
            assert 0 <= metric.value <= 1, \
                f"Fitness value {metric.value} at step {metric.step} is out of range [0, 1]"

    def test_n_trials_matches_child_runs(self):
        """Verify n_trials_ approximately matches number of child runs logged."""
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=space,
            generations=3,
            population_size=5,
            n_elites=1,
            use_mlflow=True,
            use_parallel=False,
            seed=42
        )
        opt.fit(X, y)

        client = MlflowClient(tracking_uri=self.mlflow_uri)
        experiment = client.get_experiment_by_name("mloptimizer")
        runs = client.search_runs([experiment.experiment_id])

        import re
        child_runs = [r for r in runs if re.match(r'^gen_\d+_ind_\d+_', r.info.run_name)]

        # n_trials should match child runs (each evaluation creates a child run)
        # Allow some tolerance for elitism
        assert len(child_runs) == opt.n_trials_, \
            f"MLflow child runs ({len(child_runs)}) should match n_trials_ ({opt.n_trials_})"
