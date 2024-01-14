import os
# import shutil

import pytest
from sklearn.datasets import load_iris, load_breast_cancer

from mloptimizer.genoptimizer import Hyperparam
from mloptimizer.genoptimizer import TreeOptimizer


@pytest.fixture
def default_tree_optimizer():
    X, y = load_iris(return_X_y=True)
    return TreeOptimizer(X, y)


@pytest.fixture
def default_tree_optimizer2():
    X, y = load_breast_cancer(return_X_y=True)
    return TreeOptimizer(X, y, "Optimizer2")


@pytest.fixture
def custom_params_tree_optimizer():
    custom_params = {
        "max_depth": Hyperparam("max_depth", 2, 4, int),
    }
    X, y = load_iris(return_X_y=True)
    return TreeOptimizer(X, y, custom_hyperparams=custom_params)


@pytest.fixture
def custom_fixed_params_tree_optimizer():
    fixed_params = {
        "max_depth": 4,
        "min_samples_split": 10
    }
    X, y = load_iris(return_X_y=True)
    return TreeOptimizer(X, y, custom_fixed_hyperparams=fixed_params)


@pytest.fixture
def custom_all_params_tree_optimizer():
    fixed_params = {
        "min_samples_split": 10
    }
    custom_params = {
        "max_depth": Hyperparam("max_depth", 2, 4, int),
    }
    X, y = load_iris(return_X_y=True)
    return TreeOptimizer(X, y, custom_hyperparams=custom_params, custom_fixed_hyperparams=fixed_params)


# Test vanilla TreeOptimizer
# Test custom parameters TreeOptimizer
# Test fixed parameters TreeOptimizer

def test_tree_optimizer_get_params(default_tree_optimizer):
    assert default_tree_optimizer.get_hyperparams() == default_tree_optimizer.get_default_hyperparams()


def test_custom_tree_optimizer_get_params(custom_params_tree_optimizer):
    assert custom_params_tree_optimizer.get_hyperparams() != custom_params_tree_optimizer.get_default_hyperparams()


def test_custom_fixed_tree_optimizer_get_params(custom_fixed_params_tree_optimizer):
    custom_p = custom_fixed_params_tree_optimizer.get_hyperparams()
    default_p = custom_fixed_params_tree_optimizer.get_default_hyperparams()
    assert custom_p != default_p


def test_create_tree_optimizer(default_tree_optimizer):
    assert os.path.isdir(default_tree_optimizer.get_folder()) and os.path.exists(default_tree_optimizer.get_log_file())
    # shutil.rmtree(default_tree_optimizer.get_folder())


def test_create_custom_fixed_params_tree_optimizer(custom_fixed_params_tree_optimizer):
    custom_fixed_params_tree_optimizer.optimize_clf(3, 3)


def test_create_tree_optimizer2(default_tree_optimizer2):
    assert os.path.isdir(default_tree_optimizer2.get_folder()) and os.path.exists(
        default_tree_optimizer2.get_log_file())
    # shutil.rmtree(default_tree_optimizer2.get_folder())


def test_tree_all_params_tree_optimizer(custom_all_params_tree_optimizer):
    custom_all_params_tree_optimizer.optimize_clf(3, 3)
