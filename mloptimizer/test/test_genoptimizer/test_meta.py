import pytest
from mloptimizer import SklearnOptimizer, Hyperparam
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

default_hyperparams = {
            "min_samples_split": Hyperparam("min_samples_split", 2, 50, int),
            "min_samples_leaf": Hyperparam("min_samples_leaf", 1, 20, int),
            "max_depth": Hyperparam("max_depth", 2, 20, int),
            "min_impurity_decrease": Hyperparam("min_impurity_decrease", 0, 150, float, 1000),
            "ccp_alpha": Hyperparam("ccp_alpha", 0, 300, float, 100000)
        }


def test_mloptimizer():
    X, y = load_iris(return_X_y=True)
    mlopt = SklearnOptimizer(clf_class=DecisionTreeClassifier, custom_hyperparams=default_hyperparams,
                             features=X, labels=y)
    mlopt.optimize_clf(5, 5)
    assert mlopt is not None
