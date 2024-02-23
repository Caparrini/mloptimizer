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
            'gamma': Hyperparam("gamma", 10, 100000000, float, 100),
            'max_iter': Hyperparam("max_iter", 100, 1000, int)
        }
        return default_hyperparams

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = SVC(random_state=self.mlopt_seed,
                  **individual_dict
                  )
        return clf
