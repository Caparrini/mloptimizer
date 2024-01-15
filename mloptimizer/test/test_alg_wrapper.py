import pytest
from mloptimizer.alg_wrapper import CustomXGBClassifier, generate_model
from sklearn.datasets import load_breast_cancer
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


def test_fit_pred():
    uat = KerasClassifier(build_fn=generate_model, epochs=10,
                          batch_size=5)
    X, y = load_breast_cancer(return_X_y=True)
    uat.fit(X, y)
    preds = uat.predict(X)
    assert not np.all((preds == 0))
