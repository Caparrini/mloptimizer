import unittest
from sklearn.utils.estimator_checks import check_estimator
from mloptimizer.alg_wrapper import CustomXGBClassifier


class CustomXGBTest(unittest.TestCase):
    def test_sklearn_interface(self):
        uat = CustomXGBClassifier()
        self.assertEqual(check_estimator(uat), None)


if __name__ == '__main__':
    unittest.main()
