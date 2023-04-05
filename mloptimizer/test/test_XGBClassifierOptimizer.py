import numpy as np
import pytest
import pandas as pd
import os

from mloptimizer.genoptimizer import Param
from mloptimizer.genoptimizer import XGBClassifierOptimizer
from sklearn.datasets import load_iris, load_breast_cancer
import shutil


@pytest.fixture
def default_xgb_optimizer():
    X, y = load_iris(return_X_y=True)
    return XGBClassifierOptimizer(X, y)


def test_xgb_optimizer(default_xgb_optimizer):
    default_xgb_optimizer.optimize_clf(8, 10)
