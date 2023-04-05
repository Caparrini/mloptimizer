# Import the package to test
# import mloptimizer

# Import the module
import unittest
from mloptimizer.genoptimizer import Param
from mloptimizer.genoptimizer import TreeOptimizer, MLPOptimizer, \
    SVCOptimizer, XGBClassifierOptimizer, CustomXGBClassifierOptimizer, \
    KerasClassifierOptimizer, CatBoostClassifierOptimizer
from unittest import TestCase
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import os
import shutil


class ParamTest(TestCase):
    def test_int_param(self):
        int_test = 3
        uat = Param('integer', 2, 10, int)
        self.assertEqual(int_test, uat.correct(int_test))


class XGBClassifierOptimizerTest(TestCase):
    def test_load_breast_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = XGBClassifierOptimizer(X, y)
        uat.optimize_clf(3, 3)
        # shutil.rmtree(uat.checkpoint_path)

    def test_checkpoint_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = XGBClassifierOptimizer(X, y)
        uat.optimize_clf(10, 3)
        checkpoint = os.path.join(uat.checkpoint_path, "cp_gen_2.pkl")
        uat.optimize_clf(10, 3, checkpoint=checkpoint)
        shutil.rmtree(uat.checkpoint_path)


class CatBoostClassifierOptimizerTest(TestCase):
    def test_load_iris_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = CatBoostClassifierOptimizer(X, y)
        uat.optimize_clf(3, 3)
        shutil.rmtree(uat.checkpoint_path)


class CustomXGBClassifierOptimizerTest(TestCase):
    def test_load_breast_cancer_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = CustomXGBClassifierOptimizer(X, y)
        uat.optimize_clf(3, 3)
        shutil.rmtree(uat.checkpoint_path)

    def test_custom_params(self):
        X, y = load_breast_cancer(return_X_y=True)
        custom_params = {
            'eta': Param("eta", 5, 10, float, 10),
            'gamma': Param("gamma", 10, 20, int),
            'max_depth': Param("max_depth", 10, 20, int)
        }
        uat = CustomXGBClassifierOptimizer(X, y,
                                           custom_params=custom_params)
        uat.optimize_clf(3, 3)
        shutil.rmtree(uat.checkpoint_path)

    def test_fixed_params(self):
        X, y = load_breast_cancer(return_X_y=True)
        fixed_params = {
            'obj': None,
            'feval': None
        }
        uat = CustomXGBClassifierOptimizer(X, y,
                                           custom_fixed_params=fixed_params)
        uat.optimize_clf(3, 3)
        shutil.rmtree(uat.checkpoint_path)


class KerasClassifierOptimizerTest(TestCase):
    def test_load_breast_cancer_optimizer(self):
        X, y = load_breast_cancer(return_X_y=True)
        uat = KerasClassifierOptimizer(X, y)
        uat.optimize_clf(3, 3)
        shutil.rmtree(uat.checkpoint_path)


class SCVOptimizerTest(TestCase):
    def test_load_iris_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = SVCOptimizer(X, y)
        uat.optimize_clf(10, 2)
        shutil.rmtree(uat.checkpoint_path)


class MLPOptimizerTest(TestCase):
    def test_load_iris_optimizer(self):
        X, y = load_iris(return_X_y=True)
        uat = MLPOptimizer(X, y)
        uat.optimize_clf(2, 2)
        shutil.rmtree(uat.checkpoint_path)


if __name__ == '__main__':
    unittest.main()
