import pytest
from mloptimizer.aux.plots import logbook_to_pandas, plot_logbook, plot_search_space
from mloptimizer.core import Optimizer
from mloptimizer.hyperparams import HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


@pytest.fixture
def default_tree_optimizer():
    X, y = load_iris(return_X_y=True)
    default_hyperparameter_space = HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier)
    opt = Optimizer(features=X, labels=y, estimator_class=DecisionTreeClassifier,
                    hyperparam_space=default_hyperparameter_space)
    opt.optimize_clf(10, 10)
    return opt


def test_logbook_to_pandas(default_tree_optimizer):
    logbook = default_tree_optimizer.logbook
    df = logbook_to_pandas(logbook)
    assert df is not None


def test_plot_logbook(default_tree_optimizer):
    logbook = default_tree_optimizer.logbook
    fig = plot_logbook(logbook)
    assert fig is not None


def test_plot_search_space(default_tree_optimizer):
    populations_df = default_tree_optimizer.runs[-1].population_2_df()
    fig = plot_search_space(populations_df)
    assert fig is not None
