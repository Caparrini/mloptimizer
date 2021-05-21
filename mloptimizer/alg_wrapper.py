import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


class CustomXGBClassifier(BaseEstimator):
    def __init__(self, base_score=0.5, booster="gbtree", eval_metric="auc",
                 eta=0.077, gamma=18, subsample=0.728, colsample_bylevel=1,
                 colsample_bytree=0.46, max_delta_step=0, max_depth=7,
                 min_child_weight=1, seed=1, alpha=0, scale_pos_weight=4.43):
        self.base_score = base_score
        self.booster = booster
        self.eval_metric = eval_metric
        self.eta = eta
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bytree = colsample_bytree
        self.max_delta_step = max_delta_step
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.seed = seed
        self.alpha = alpha
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y):
        check_array(X)
        check_array(y, ensure_2d=False)
        dtrain = xgb.DMatrix(X, y)
        params = self.get_params()
        # Workaround: lambda is a reserved word in Python
        params['lambda'] = 1
        self._xclf = xgb.train(params, dtrain, num_boost_round=50)
        #                      obj=profit_objective_f, feval=saving_f_tp)
        return self

    def predict(self, X):
        check_array(X)
        dtest = xgb.DMatrix(X)
        return self._xclf.predict(dtest)
