from sklearn.preprocessing import MinMaxScaler

from mloptimizer.eda import read_dataset
from mloptimizer.genoptimizer import XGBClassifierOptimizer, MLPOptimizer, SVCOptimizer
from xgboost import XGBClassifier
import mloptimizer.miscellaneous
from mloptimizer.model_evaluation import KFoldStratifiedAccuracy
import logging
from sklearn.neural_network import MLPClassifier


def get_clf():
    return MLPClassifier()


def time_estimation():
    totals = [579, 1511, 882, 1993, 3288, 5178, 7359, 9228, 12493, 18523, 34844, 53374, 81430, 99469, 123633,
              162745, 212800, 174879, 118216, 91358, 77942, 40973, 15338]

    for t in totals:
        logging.info("DATASET CON {} ELEMENTOS".format(t))
        X, y = read_dataset("data_sample_train.csv", ohe=False, scaler=MinMaxScaler(), return_x_y=True,
                            vars_type=['numerical'], sampling=t, replace=True)
        #uat = XGBClassifierOptimizer(x, y, "file")
        #uat.optimize_clf(2, 2)
        clf = get_clf()

        KFoldStratifiedAccuracy(X, y, clf, 4, random_state=1)




if __name__ == "__main__":
    mloptimizer.miscellaneous.init_logger('time_estimation.log')
    time_estimation()
