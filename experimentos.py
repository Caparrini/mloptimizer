from sklearn.preprocessing import MinMaxScaler

from mloptimizer.eda import read_dataset
from mloptimizer.genoptimizer import XGBClassifierOptimizer, MLPOptimizer, SVCOptimizer


def exp1():
    x, y = read_dataset("data.balanced.csv", ohe=0, scaler=MinMaxScaler(), samples=20000, return_X_y=True)
    uat = XGBClassifierOptimizer(x, y, "file")
    uat.optimize_clf(10, 2)
    uat = MLPOptimizer(x, y, "file")
    uat.optimize_clf(10, 2)
    uat = SVCOptimizer(x, y, "file")
    uat.optimize_clf(10, 2)

if __name__ == "__main__":
    exp1()
