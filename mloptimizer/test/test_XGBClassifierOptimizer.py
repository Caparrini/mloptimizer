import pytest
from sklearn.datasets import load_iris

from mloptimizer.genoptimizer import Param
from mloptimizer.genoptimizer import XGBClassifierOptimizer


@pytest.fixture
def default_xgb_optimizer():
    X, y = load_iris(return_X_y=True)
    return XGBClassifierOptimizer(X, y)


@pytest.fixture
def custom_params_xgb_optimizer():
    custom_params = {
        "max_depth": Param("max_depth", 2, 4, int),
    }
    X, y = load_iris(return_X_y=True)
    return XGBClassifierOptimizer(X, y, custom_params=custom_params)


@pytest.fixture
def custom_fixed_params_xgb_optimizer():
    fixed_params = {
        "max_depth": 4,
        "gamma": 10
    }
    X, y = load_iris(return_X_y=True)
    return XGBClassifierOptimizer(X, y, custom_fixed_params=fixed_params)


@pytest.fixture
def custom_all_params_xgb_optimizer():
    fixed_params = {
        "gamma": 0
    }
    custom_params = {
        "max_depth": Param("max_depth", 2, 10, int),
    }
    X, y = load_iris(return_X_y=True)
    return XGBClassifierOptimizer(X, y, custom_params=custom_params, custom_fixed_params=fixed_params)


def test_xgb_optimizer(default_xgb_optimizer):
    default_xgb_optimizer.optimize_clf(4, 4)


def test_custom_xgb_optimizer(custom_params_xgb_optimizer):
    custom_params_xgb_optimizer.optimize_clf(4, 4)


def test_custom_fixed_xgb_optimizer(custom_fixed_params_xgb_optimizer):
    custom_fixed_params_xgb_optimizer.optimize_clf(4, 4)


def test_custom_all_xgb_optimizer(custom_all_params_xgb_optimizer):
    custom_all_params_xgb_optimizer.optimize_clf(4, 4)

