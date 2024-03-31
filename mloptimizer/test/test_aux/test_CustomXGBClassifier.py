import pytest
import numpy as np
from sklearn.datasets import make_classification
from mloptimizer.aux.alg_wrapper import CustomXGBClassifier


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    return X, y


def test_custom_xgb_classifier_init():
    classifier = CustomXGBClassifier()
    assert classifier.base_score == 0.5
    assert classifier.booster == "gbtree"
    # ... continue for other attributes


def test_custom_xgb_classifier_fit(sample_data):
    X, y = sample_data
    classifier = CustomXGBClassifier()
    classifier.fit(X, y)
    assert hasattr(classifier, "_xclf")


def test_custom_xgb_classifier_predict(sample_data):
    X, y = sample_data
    classifier = CustomXGBClassifier()
    classifier.fit(X, y)
    preds = classifier.predict(X)
    assert len(preds) == len(y)
    assert np.array_equal(preds, preds.astype(bool))


def test_custom_xgb_classifier_predict_proba(sample_data):
    X, y = sample_data
    classifier = CustomXGBClassifier()
    classifier.fit(X, y)
    proba = classifier.predict_proba(X)
    assert len(proba) == len(y)
    assert np.all(proba >= 0) and np.all(proba <= 1)  # Probabilities between 0 and 1
