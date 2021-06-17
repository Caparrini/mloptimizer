# Import the package to test
# import mloptimizer

# Import the module
import unittest
from mloptimizer.genoptimizer import Param
from mloptimizer.genoptimizer import TreeOptimizer, MLPOptimizer, \
    SVCOptimizer, XGBClassifierOptimizer, CustomXGBClassifierOptimizer, \
    KerasClassifierOptimizer
from mloptimizer.eda import read_dataset
from unittest import TestCase
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


class ParamTest(TestCase):
    def test_int_param(self):
        int_test = 3
        uat = Param('integer', 2, 10, int)
        self.assertEqual(int_test, uat.correct(int_test))


class XGBClassifierOptimizerTest(TestCase):
    def test_load_boston_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = XGBClassifierOptimizer(X, y, "file")
        uat.optimize_clf(3, 3)


class CustomXGBClassifierOptimizerTest(TestCase):
    def test_load_breast_cancer_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = CustomXGBClassifierOptimizer(X, y, "file")
        uat.optimize_clf(3, 3)


class KerasClassifierOptimizerTest(TestCase):
    def test_load_breast_cancer_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = KerasClassifierOptimizer(X, y, "file")
        uat.optimize_clf(3, 3)


class TreeOptimizerTest(TestCase):
    def test_default_params_basic_optimizer(self):
        features = [[0, 1, 2, 3], [0, 1, 1, 2]]
        labels = [1, 2]
        uat = TreeOptimizer(features, labels)
        self.assertEqual(uat.get_default_params(), uat.get_params())

    def test_default_params_basic_optimizer(self):
        features = [[0, 1, 2, 3], [0, 1, 1, 2]]
        labels = [1, 2]
        uat = TreeOptimizer(features, labels, "file", {"min_samples_split": Param("min_samples_split", 100, 200, int)})
        self.assertNotEqual(uat.get_default_params(), uat.get_params())

    def test_load_boston_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = TreeOptimizer(X, y, "file")
        uat.optimize_clf(2, 2)

    def test_load_breast_cancer_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = TreeOptimizer(X, y, "file")
        uat.optimize_clf(10, 10)


class SCVOptimizerTest(TestCase):
    def test_load_boston_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = SVCOptimizer(X, y, "file")
        uat.optimize_clf(10, 2)


class MLPOptimizerTest(TestCase):
    def test_load_boston_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = MLPOptimizer(X, y, "file")
        uat.optimize_clf(2, 2)


class PaperTest(TestCase):
    def test_experiment_1(self):
        x, y = read_dataset("data_sample_train.csv", ohe=0, scaler=MinMaxScaler(), samples=1000, return_x_y=True)
        uat = XGBClassifierOptimizer(x, y, "file")
        uat.optimize_clf(2, 2)
        uat = MLPOptimizer(x, y, "file")
        uat.optimize_clf(2, 2)
        uat = SVCOptimizer(x, y, "file")
        uat.optimize_clf(2, 2)


if __name__ == '__main__':
    unittest.main()