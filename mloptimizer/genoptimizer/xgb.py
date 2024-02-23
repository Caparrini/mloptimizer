from abc import ABC
import xgboost as xgb

from mloptimizer.genoptimizer import Hyperparam, BaseOptimizer
from mloptimizer.alg_wrapper import CustomXGBClassifier


class XGBClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from xgboost.XGBClassifier.
    It inherits from BaseOptimizer.
    """

    @staticmethod
    def get_default_hyperparams():
        default_hyperparams = {
            'colsample_bytree': Hyperparam("colsample_bytree", 3, 10, float, 10),
            'gamma': Hyperparam("gamma", 0, 20, int),
            'learning_rate': Hyperparam("learning_rate", 1, 100, float, 1000),
            'max_depth': Hyperparam("max_depth", 3, 20, int),
            'n_estimators': Hyperparam("n_estimators", 100, 500, int),
            'subsample': Hyperparam("subsample", 700, 1000, float, 1000),
            'scale_pos_weight': Hyperparam("scale_pos_weight", 15, 40, float, 100)
        }
        return default_hyperparams

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = xgb.XGBClassifier(seed=self.mlopt_seed,
                                random_state=self.mlopt_seed,
                                **individual_dict
                                )
        return clf


class CustomXGBClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from alg_wrapper.CustomXGBClassifier.
    It inherits from BaseOptimizer.
    """

    @staticmethod
    def get_default_hyperparams():
        default_hyperparams = {
            'eta': Hyperparam("eta", 0, 100, float, 100),
            'colsample_bytree': Hyperparam("colsample_bytree", 3, 10, float, 10),
            'alpha': Hyperparam("alpha", 0, 100, float, 100),
            'lambda': Hyperparam("lambda", 0, 100, float, 100),
            'gamma': Hyperparam("gamma", 0, 100, float, 100),
            'max_depth': Hyperparam("max_depth", 3, 14, int),
            'subsample': Hyperparam("subsample", 70, 100, float, 100),
            'num_boost_round': Hyperparam("num_boost_round", 2, 100, int),
            'scale_pos_weight': Hyperparam("scale_pos_weight", 10, 10000, float, 100),
            'min_child_weight': Hyperparam("min_child_weight", 0, 100, float, 10)
        }
        return default_hyperparams

    def get_default_fixed_hyperparams(self):
        default_fixed_hyperparams = {
            'obj': None,
            'feval': None
        }
        return default_fixed_hyperparams

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = CustomXGBClassifier(base_score=0.5,
                                  booster="gbtree",
                                  eval_metric="auc",
                                  eta=individual_dict['eta'],
                                  gamma=individual_dict['gamma'],
                                  subsample=individual_dict['subsample'],
                                  colsample_bylevel=1,
                                  colsample_bytree=individual_dict['colsample_bytree'],
                                  max_delta_step=0,
                                  max_depth=individual_dict['max_depth'],
                                  min_child_weight=individual_dict['min_child_weight'],
                                  seed=self.mlopt_seed,
                                  alpha=individual_dict['alpha'],
                                  reg_lambda=individual_dict['lambda'],
                                  num_boost_round=individual_dict['num_boost_round'],
                                  scale_pos_weight=individual_dict['scale_pos_weight'],
                                  obj=self.fixed_hyperparams['obj'],
                                  feval=self.fixed_hyperparams['feval'])
        return clf
