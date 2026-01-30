import socket
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking.client import MlflowClient
from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

MLFLOW_PORT = 5051  # Using non-standard port to avoid conflicts
MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"


def is_port_open(port, timeout=1):
    """Check if a port is open and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session")
def mlflow_server():
    """Fixture for MLflow server tests.

    This fixture does NOT start a server automatically.
    Tests that use this fixture should check server availability
    and skip if not available.

    To run these tests, start an MLflow server manually:
        mlflow server --port 5001
    """
    yield None


def test_genetic_search_creates_mlflow_runs(mlflow_server):
    """Test MLflow integration with a remote server.

    This test requires an MLflow server running on port 5001.
    It will be skipped if the server is not available.

    To run this test manually:
        1. Start MLflow server: mlflow server --port 5001
        2. Run: pytest mloptimizer/test/interfaces/api/test_genetic_search_mlflow_tracking.py -v
    """
    # Skip if server not available (check with short timeout)
    if not is_port_open(MLFLOW_PORT, timeout=1):
        pytest.skip(
            f"MLflow server not available at {MLFLOW_TRACKING_URI}. "
            f"Start with: mlflow server --port {MLFLOW_PORT}"
        )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Prepare dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Build hyperparameter space
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Run search with MLflow logging
    search = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        scoring="accuracy",
        **{"generations": 2, "population_size": 3},
        use_mlflow=True
    )
    search.fit(X_train, y_train)

    # Check MLflow logs
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiments = client.search_experiments()
    assert experiments, "No experiments found"

    # Find the latest experiment (assumes recent test run)
    latest_exp = sorted(experiments, key=lambda e: e.creation_time, reverse=True)[0]
    runs = client.search_runs([latest_exp.experiment_id])
    assert runs, "No runs found"

    # Find parent runs (name starts with YYYYMMDD_HHMMSS_)
    import re
    parent_runs = [r for r in runs if re.match(r'^\d{8}_\d{6}_', r.info.run_name)]
    child_runs = [r for r in runs if re.match(r'^gen_\d+_ind_\d+_', r.info.run_name)]

    assert parent_runs, "No parent MLflow run created"
    assert len(parent_runs) >= 1, f"Expected at least 1 parent run, found {len(parent_runs)}"

    # Verify parent run has expected data
    parent = parent_runs[0]
    assert parent.data.params, "No parameters logged in parent run"
    assert "population_size" in parent.data.params, "population_size not logged"
    assert "generations" in parent.data.params, "generations not logged"

    # Verify generation-level metrics
    assert "generation_best_fitness" in parent.data.metrics, "generation_best_fitness not logged"
    assert "generation_avg_fitness" in parent.data.metrics, "generation_avg_fitness not logged"
    assert "final_best_fitness" in parent.data.metrics, "final_best_fitness not logged"

    # Verify tags
    assert "estimator_class" in parent.data.tags, "estimator_class tag not set"
    assert "dataset_samples" in parent.data.tags, "dataset_samples tag not set"

    # If child runs exist, verify their structure
    if child_runs:
        child = child_runs[0]
        assert child.data.params, "No parameters logged in child run"
        assert child.data.metrics, "No metrics logged in child run"
        assert "gen_" in child.info.run_name, "Child run name format incorrect"
