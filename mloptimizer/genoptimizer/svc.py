from abc import ABC
from sklearn.svm import SVC

from mloptimizer.genoptimizer import Hyperparam, BaseOptimizer


class SVCOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a support vector machine classifier from sklearn.svm.SVC.
    It inherits from BaseOptimizer.
    """

    @staticmethod
    def get_default_hyperparams():
        default_hyperparams = {
            'C': Hyperparam("C", 1, 10000, float, 10),
            'degree': Hyperparam("degree", 0, 6, int),
            'gamma': Hyperparam("gamma", 10, 100000000, float, 100)
        }
        return default_hyperparams

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = SVC(C=individual_dict['C'],
                  cache_size=8000000,
                  class_weight="balanced",
                  coef0=0.0,
                  decision_function_shape='ovr',
                  degree=individual_dict['degree'], gamma=individual_dict['gamma'],
                  kernel='rbf',
                  max_iter=100000,
                  probability=False,
                  random_state=None,
                  shrinking=True,
                  tol=0.001,
                  verbose=False
                  )
        return clf
