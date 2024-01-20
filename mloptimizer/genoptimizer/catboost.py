from abc import ABC
from catboost import CatBoostClassifier

from mloptimizer.genoptimizer import Hyperparam, BaseOptimizer


class CatBoostClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from catboost.CatBoostClassifier.
    It inherits from BaseOptimizer.
    """

    @staticmethod
    def get_default_hyperparams():
        default_hyperparams = {
            'eta': Hyperparam("eta", 1, 10, float, 10),
            'max_depth': Hyperparam("max_depth", 3, 16, int),  # Max is 16
            'n_estimators': Hyperparam("n_estimators", 100, 500, int),
            'subsample': Hyperparam("subsample", 700, 1000, float, 1000),
        }
        return default_hyperparams

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = CatBoostClassifier(
            **individual_dict, auto_class_weights="Balanced",
            bootstrap_type='Bernoulli'
        )
        return clf
