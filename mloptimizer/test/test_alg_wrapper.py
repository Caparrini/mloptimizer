import unittest
from sklearn.utils.estimator_checks import check_estimator
from mloptimizer.alg_wrapper import CustomXGBClassifier, generate_model
from sklearn.datasets import load_breast_cancer
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


class CustomXGBTest(unittest.TestCase):
    def test_sklearn_interface(self):
        uat = CustomXGBClassifier()
        self.assertEqual(check_estimator(uat), None)


class CustomKerasClassifier(unittest.TestCase):

    def test_fit_pred(self):
        uat = KerasClassifier(build_fn=generate_model, epochs=10,
                              batch_size=5)
        X, y = load_breast_cancer(return_X_y=True)
        uat.fit(X, y)
        preds = uat.predict(X)
        self.assertFalse(np.all((preds == 0)))


if __name__ == '__main__':
    unittest.main()
