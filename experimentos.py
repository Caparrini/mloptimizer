from sklearn.preprocessing import MinMaxScaler

from mloptimizer.eda import read_dataset
from mloptimizer.genoptimizer import XGBClassifierOptimizer, MLPOptimizer, SVCOptimizer


def exp1():
    x, y = read_dataset("data_sample_train.csv", ohe=False, scaler=MinMaxScaler(), return_x_y=True)
    uat = XGBClassifierOptimizer(x, y, "file")
    uat.optimize_clf(100, 20)


if __name__ == "__main__":
    exp1()
