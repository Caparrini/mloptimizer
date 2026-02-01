"""
Performance Regression Tests
============================

These tests ensure that optimization performance hasn't regressed.
They run early in the test suite to catch performance issues quickly.

The time limits are set conservatively to avoid flaky tests while still
catching significant regressions (e.g., thread contention issues).
"""
import time
import pytest
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder


# Maximum allowed time in seconds for XGBoost optimization
# This catches thread contention issues that can cause 10-100x slowdowns
XGBOOST_MAX_TIME_SECONDS = 30


class TestPerformanceRegression:
    """Performance regression tests - run these first to catch slowdowns early."""

    def test_xgboost_optimization_time(self):
        """
        Test that XGBoost optimization completes within acceptable time.

        This test catches thread contention issues that previously caused
        tests to take 40+ seconds instead of <5 seconds. If this test fails,
        check:
        - n_jobs=1 is set in XGBoost default hyperparameter space
        - parallel_config is properly limiting inner threads
        """
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

        opt = GeneticSearch(
            estimator_class=XGBClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            seed=42,
            use_parallel=False
        )

        start_time = time.time()
        opt.fit(X, y)
        elapsed_time = time.time() - start_time

        assert elapsed_time < XGBOOST_MAX_TIME_SECONDS, (
            f"XGBoost optimization took {elapsed_time:.2f}s, "
            f"expected < {XGBOOST_MAX_TIME_SECONDS}s. "
            "Possible thread contention issue - check n_jobs=1 in config."
        )

        # Verify optimization completed successfully
        assert opt.best_estimator_ is not None
        assert opt.best_params_ is not None

    def test_xgboost_parallel_optimization_time(self):
        """
        Test that parallel XGBoost optimization completes within acceptable time.

        Parallel mode should not significantly increase total time compared
        to sequential mode when thread contention is properly managed.
        """
        X, y = load_iris(return_X_y=True)
        space = HyperparameterSpaceBuilder.get_default_space(XGBClassifier)

        opt = GeneticSearch(
            estimator_class=XGBClassifier,
            hyperparam_space=space,
            generations=2,
            population_size=4,
            seed=42,
            use_parallel=True
        )

        start_time = time.time()
        opt.fit(X, y)
        elapsed_time = time.time() - start_time

        # Parallel mode has some overhead but shouldn't be drastically slower
        max_parallel_time = XGBOOST_MAX_TIME_SECONDS * 1.5

        assert elapsed_time < max_parallel_time, (
            f"Parallel XGBoost optimization took {elapsed_time:.2f}s, "
            f"expected < {max_parallel_time:.1f}s. "
            "Check parallel_config and inner_max_num_threads settings."
        )

        assert opt.best_estimator_ is not None
