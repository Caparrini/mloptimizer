"""
Pytest configuration for mloptimizer tests.
"""
import pytest


# =============================================================================
# Session-scoped fixtures for commonly used data (loaded once per test session)
# =============================================================================

@pytest.fixture(scope="session")
def iris_data():
    """Session-scoped iris dataset to avoid repeated loading."""
    from sklearn.datasets import load_iris
    return load_iris(return_X_y=True)


@pytest.fixture(scope="session")
def breast_cancer_data():
    """Session-scoped breast cancer dataset to avoid repeated loading."""
    from sklearn.datasets import load_breast_cancer
    return load_breast_cancer(return_X_y=True)


@pytest.fixture(scope="session")
def diabetes_data():
    """Session-scoped diabetes dataset to avoid repeated loading."""
    from sklearn.datasets import load_diabetes
    return load_diabetes(return_X_y=True)


# =============================================================================
# Test ordering
# =============================================================================

def pytest_collection_modifyitems(items):
    """
    Reorder tests to run performance regression tests first.

    This ensures that performance issues are caught early in the test run,
    before spending time on other tests that might be affected by the same issue.
    """
    performance_tests = []
    other_tests = []

    for item in items:
        if "test_00_performance_regression" in str(item.fspath):
            performance_tests.append(item)
        else:
            other_tests.append(item)

    # Performance tests run first, then everything else in original order
    items[:] = performance_tests + other_tests
