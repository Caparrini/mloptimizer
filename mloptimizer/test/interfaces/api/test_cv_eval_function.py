import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder
from mloptimizer.domain.evaluation import kfold_stratified_score


@pytest.fixture
def iris_data():
    """Fixture for Iris dataset."""
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_rf():
    """Basic hyperparameter space for RF."""
    builder = HyperparameterSpaceBuilder()
    return (builder
            .add_int_param("n_estimators", 50, 50)  # fixed for consistency
            .add_int_param("max_depth", 6, 6)
            .build())


def test_eval_function_vs_cv_equivalence(iris_data, hyperparam_space_rf):
    X, y = iris_data
    seed = 123
    n_splits = 4

    # First: use old-style eval_function
    search_eval = GeneticSearch(
        estimator_class=RandomForestClassifier,
        hyperparam_space=hyperparam_space_rf,
        genetic_params_dict={"generations": 30, "population_size": 30},
        eval_function=lambda features, labels, clf, metrics, random_state=None:
            kfold_stratified_score(features, labels, clf, metrics, n_splits=n_splits, random_state=seed),
        seed=seed,
        use_parallel=False
    )
    search_eval.fit(X, y)
    result_eval_fn = search_eval.cv_results_

    # Second: use cv object (new-style)
    stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    search_cv = GeneticSearch(
        estimator_class=RandomForestClassifier,
        hyperparam_space=hyperparam_space_rf,
        genetic_params_dict={"generations": 30, "population_size": 30},
        cv=stratified_cv,
        seed=seed,
        use_parallel=False
    )
    search_cv.fit(X, y)
    result_cv_obj = search_cv.cv_results_

    # Assert the results are numerically equivalent
    for ind1, ind2 in zip(result_eval_fn, result_cv_obj):
        for key in ind1:
            assert np.isclose(ind1[key], ind2[key]), f"Mismatch in {key}: {ind1[key]} != {ind2[key]}"
