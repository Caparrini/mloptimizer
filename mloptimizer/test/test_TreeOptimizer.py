import numpy as np
import pytest
import pandas as pd
import os

from mloptimizer.genoptimizer import Param
from mloptimizer.genoptimizer import TreeOptimizer
from sklearn.datasets import load_iris, load_breast_cancer
import shutil


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
        "max_depth": Param("max_depth", 2, 4, int),
    }
    X, y = load_iris(return_X_y=True)
    return TreeOptimizer(X, y, custom_params=custom_params)


@pytest.fixture
def custom_fixed_params_tree_optimizer():
    fixed_params = {
        "max_depth": 4,
    }
    X, y = load_iris(return_X_y=True)
    return TreeOptimizer(X, y, custom_fixed_params=fixed_params)


# Test vanilla TreeOptimizer
# Test custom parameters TreeOptimizer
# Test fixed parameters TreeOptimizer

def test_tree_optimizer_get_params(default_tree_optimizer):
    assert default_tree_optimizer.get_params() == default_tree_optimizer.get_default_params()


def test_custom_tree_optimizer_get_params(custom_params_tree_optimizer):
    assert custom_params_tree_optimizer.get_params() != custom_params_tree_optimizer.get_default_params()


def test_custom_fixed_tree_optimizer_get_params(custom_fixed_params_tree_optimizer):
    assert custom_fixed_params_tree_optimizer.get_params() != custom_fixed_params_tree_optimizer.get_default_params()


def test_create_tree_optimizer(default_tree_optimizer):
    assert os.path.isdir(default_tree_optimizer.get_folder()) and os.path.exists(default_tree_optimizer.get_log_file())
    shutil.rmtree(default_tree_optimizer.get_folder())


def test_create_tree_optimizer2(default_tree_optimizer2):
    assert os.path.isdir(default_tree_optimizer2.get_folder()) and os.path.exists(
        default_tree_optimizer2.get_log_file())
    shutil.rmtree(default_tree_optimizer2.get_folder())


def test_tree_optimizer_get_params(default_tree_optimizer):
    default_tree_optimizer.optimize_clf(8, 10)
