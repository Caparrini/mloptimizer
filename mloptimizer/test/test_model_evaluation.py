from mloptimizer.evaluation import kfold_stratified_score, temporal_kfold_score, \
    train_score, train_test_score, kfold_score
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture
def classification_mock_data():
    # Create mock features and labels for classification testing
    features, labels = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    return features, labels


def test_kfold_stratified_score(classification_mock_data):
    features, labels = classification_mock_data
    clf = DecisionTreeClassifier()
    score = kfold_stratified_score(features, labels, clf)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # Score should be between 0 and 1


def test_temporal_kfold_score(classification_mock_data):
    features, labels = classification_mock_data
    clf = DecisionTreeClassifier()
    score = temporal_kfold_score(features, labels, clf)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # Score should be between 0 and 1


def test_train_score(classification_mock_data):
    features, labels = classification_mock_data
    clf = DecisionTreeClassifier()
    accuracy = train_score(features, labels, clf, score_function=accuracy_score)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_test_train_score(classification_mock_data):
    features, labels = classification_mock_data
    clf = DecisionTreeClassifier()
    accuracy = train_test_score(features, labels, clf, score_function=accuracy_score)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_kfold_score(classification_mock_data):
    features, labels = classification_mock_data
    clf = DecisionTreeClassifier()
    accuracy = kfold_score(features, labels, clf, score_function=accuracy_score)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
