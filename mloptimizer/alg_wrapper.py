import keras
import numpy as np
import xgboost as xgb
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


class CustomXGBClassifier(BaseEstimator):
    def __init__(self, base_score=0.5, booster="gbtree", eval_metric="auc",
                 eta=0.077, gamma=18, subsample=0.728, colsample_bylevel=1,
                 colsample_bytree=0.46, max_delta_step=0, max_depth=7,
                 min_child_weight=1, seed=1, alpha=0, reg_lambda=1, scale_pos_weight=4.43,
                 obj=None, feval=None, num_boost_round=50):
        self.base_score = base_score
        self.booster = booster
        self.eval_metric = eval_metric
        self.eta = eta
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bytree = colsample_bytree
        self.max_delta_step = max_delta_step
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.seed = seed
        self.alpha = alpha
        self.scale_pos_weight = round(1/scale_pos_weight, 3)
        self.obj = obj
        self.feval = feval
        #self.tree_method = "gpu_hist"
        self.num_boost_round= num_boost_round

    def fit(self, X, y):
        check_array(X)
        check_array(y, ensure_2d=False)
        dtrain = xgb.DMatrix(X, y)
        params = self.get_params()
        # Workaround: lambda is a reserved word in Python
        params['lambda'] = params.pop("reg_lambda")
        n_boost_round = params.pop("num_boost_round")
        # Extract params for obj and feval
        if self.obj is not None:
            obj = params.pop('obj')
            feval = params.pop('feval')
            self._xclf = xgb.train(params, dtrain, num_boost_round=n_boost_round, obj=obj, feval=feval)
        else:
            params['objective'] = "binary:logistic"
            self._xclf = xgb.train(params, dtrain, num_boost_round=n_boost_round)

        return self

    def predict(self, X):
        #check_array(X)
        #dtest = xgb.DMatrix(X)
        #preds = np.array(self._xclf.predict(dtest, ntree_limit=self._xclf.best_iteration))
        p = self.predict_proba(X)
        preds = p > 0.5
        preds = preds.astype(int)
        return preds

    def predict_proba(self, X):
        if self.obj is None:
            p = self.predict_z(X)
        else:
            zs = self.predict_z(X)
            p = 1.0 / (1.0 + np.exp(-zs))
        return p

    def predict_z(self, X):
        check_array(X)
        dtest = xgb.DMatrix(X)
        preds = np.array(self._xclf.predict(dtest, ntree_limit=self._xclf.best_iteration))
        preds = preds > 0.5
        preds = preds.astype(int)
        return preds


def generate_model(learning_rate=0.01, layer_1=100, layer_2=50,
                   dropout_rate_1=0, dropout_rate_2=0):
    model = Sequential()
    model.add(Dense(layer_1, activation="relu"))
    model.add(Dropout(dropout_rate_1))
    model.add(Dense(layer_2, activation="relu"))
    model.add(Dropout(dropout_rate_2))
    model.add(Dense(1, activation="sigmoid"))

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model
