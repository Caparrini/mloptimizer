import pytest
from mloptimizer.alg_wrapper import generate_model
from sklearn.datasets import load_breast_cancer
import numpy as np
import sys, subprocess


def test_keras_classifier():
    try:
        from keras.wrappers.scikit_learn import KerasClassifier
    except ImportError as e:
        print(f"{e}: Keras is not installed. It will be installed to test.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras==2.12.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.12.1"])
        from keras.wrappers.scikit_learn import KerasClassifier

    uat = KerasClassifier(build_fn=generate_model, epochs=10,
                          batch_size=5)
    X, y = load_breast_cancer(return_X_y=True)
    uat.fit(X, y)
    preds = uat.predict(X)
    assert not np.all((preds == 0))
