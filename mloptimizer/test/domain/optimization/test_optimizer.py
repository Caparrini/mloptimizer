import pytest
import warnings
from mloptimizer.domain.optimization import Optimizer
from mloptimizer.domain.hyperspace import Hyperparam, HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from xgboost import XGBClassifier
from mloptimizer.domain.evaluation import kfold_score, train_score, train_test_score
import time
import os
from sklearn.metrics import accuracy_score

# Representative estimators for testing (covers sklearn native, ensemble, and external library)
REPRESENTATIVE_ESTIMATORS = (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)


@pytest.fixture
def default_metrics_dict():
    return {
        "accuracy": accuracy_score,
    }


@pytest.mark.parametrize('estimator_class', REPRESENTATIVE_ESTIMATORS)
def test_sklearn_optimizer(estimator_class, tmp_path):
    """Test basic optimizer creation and optimization flow."""
    x, y = load_iris(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    mlopt = Optimizer(estimator_class=estimator_class,
                      hyperparam_space=evolvable_hyperparams,
                      features=x, labels=y, folder=tmp_path)
    mlopt.optimize_clf(2, 4)
    assert mlopt is not None


@pytest.mark.parametrize('use_mlflow', [True, False])
def test_mloptimizer(use_mlflow, tmp_path):
    """Test optimization with and without MLflow."""
    x, y = load_iris(return_X_y=True)
    mlopt = Optimizer(estimator_class=DecisionTreeClassifier,
                      hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier),
                      genetic_params={"population_size": 4, "generations": 2},
                      features=x, labels=y, use_mlflow=use_mlflow, folder=tmp_path)
    mlopt.optimize_clf(4, 2)
    assert mlopt is not None


def test_checkpoints():
    """Test checkpoint saving and resuming."""
    x, y = load_iris(return_X_y=True)
    mlopt = Optimizer(estimator_class=DecisionTreeClassifier,
                      hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier),
                      genetic_params={"population_size": 4, "generations": 2},
                      features=x, labels=y, disable_file_output=False)
    clf = mlopt.optimize_clf(4, 2)

    mlopt2 = Optimizer(estimator_class=DecisionTreeClassifier,
                       hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier),
                       genetic_params={"population_size": 4, "generations": 2},
                       features=x, labels=y,
                       seed=mlopt.mlopt_seed)

    checkpoint = os.path.join(mlopt.tracker.opt_run_checkpoint_path,
                              os.listdir(mlopt.tracker.opt_run_checkpoint_path)[-2]
                              )
    clf2 = mlopt2.optimize_clf(4, 2,
                               checkpoint=checkpoint)
    assert mlopt is not None
    assert str(clf) == str(clf2)


@pytest.mark.parametrize('estimator_class', REPRESENTATIVE_ESTIMATORS)
def test_optimizer(estimator_class, default_metrics_dict, tmp_path):
    """Test optimizer with different estimators."""
    x, y = load_iris(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    opt = Optimizer(features=x, labels=y, fitness_score="accuracy",
                    metrics=default_metrics_dict, estimator_class=estimator_class,
                    genetic_params={"generations": 2, "population_size": 4}, seed=42,
                    hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    clf = opt.optimize_clf(4, 2)
    assert clf is not None


@pytest.mark.slow
def test_optimizer_use_parallel(default_metrics_dict, tmp_path):
    """Test parallel execution produces same results as sequential."""
    x, y = load_iris(return_X_y=True)
    estimator_class = DecisionTreeClassifier
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    my_seed = 25
    population = 5
    generations = 3

    opt_with_parallel = Optimizer(features=x, labels=y, fitness_score="accuracy",
                                  metrics=default_metrics_dict,
                                  seed=my_seed, use_parallel=True,
                                  hyperparam_space=evolvable_hyperparams, estimator_class=estimator_class,
                                  genetic_params={"population_size": population, "generations": generations},
                                  folder=tmp_path)

    start_time_parallel = time.time()
    clf_with_parallel = opt_with_parallel.optimize_clf(population, generations)
    end_time_parallel = time.time()

    opt = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                    seed=my_seed, use_parallel=False,
                    hyperparam_space=evolvable_hyperparams, estimator_class=estimator_class,
                    genetic_params={"population_size": population, "generations": generations},
                    folder=tmp_path)

    start_time = time.time()
    clf = opt.optimize_clf(population, generations)
    end_time = time.time()

    elapsed_time_parallel = end_time_parallel - start_time_parallel
    elapsed_time = end_time - start_time
    speedup = round(((elapsed_time_parallel / elapsed_time) - 1) * 100, 2)

    print(f"Elapsed time with parallel: {elapsed_time_parallel}")
    print(f"Elapsed time without parallel: {elapsed_time}")
    print(f"Speedup: {speedup}%")
    assert str(clf) == str(clf_with_parallel)
    if elapsed_time_parallel < elapsed_time:
        warnings.warn(
            f"Sequential execution time ({elapsed_time:.2f}s) is greater than or equal "
            f"to parallel execution time ({elapsed_time_parallel:.2f}s)."
        )


@pytest.mark.parametrize('estimator_class', REPRESENTATIVE_ESTIMATORS)
@pytest.mark.parametrize('target_score', (kfold_score, train_score, train_test_score))
def test_reproducibility(estimator_class, target_score, default_metrics_dict, tmp_path):
    """Test that same seed produces same results, different seed produces different results."""
    x, y = load_iris(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    population = 4
    generations = 2
    seed = 25
    distinct_seed = 2

    optimizer1 = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=seed, estimator_class=estimator_class,
                           genetic_params={"generations": generations, "population_size": population},
                           hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    result1 = optimizer1.optimize_clf(population_size=population, generations=generations)

    optimizer2 = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=seed, estimator_class=estimator_class,
                           genetic_params={"generations": generations, "population_size": population},
                           hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    result2 = optimizer2.optimize_clf(population_size=population, generations=generations)

    optimizer3 = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=distinct_seed, estimator_class=estimator_class,
                           genetic_params={"generations": generations, "population_size": population},
                           hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    result3 = optimizer3.optimize_clf(population_size=population, generations=generations)

    assert str(result1) == str(result2)
    assert str(result1) != str(result3)


@pytest.mark.slow
def test_custom_svc():
    """Test optimization with custom SVC hyperparameter space."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    x, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    custom_svc_evolvable_hyperparams = {
        "C": Hyperparam("C", -2, 2, 'nexp'),
        "kernel": Hyperparam("kernel", 0, 3, 'list',
                             values_str=['linear', 'poly', 'rbf', 'sigmoid']),
        "degree": Hyperparam("degree", 0, 6, 'int'),
        "gamma": Hyperparam("gamma", -3, 3, 'nexp'),
        "max_iter": Hyperparam("max_iter", 1000, 10000, 'int')
    }
    evolvable_hyperparams = HyperparameterSpace(evolvable_hyperparams=custom_svc_evolvable_hyperparams,
                                                fixed_hyperparams={}
                                                )
    opt = Optimizer(features=x, labels=y, fitness_score="balanced_accuracy",
                    metrics={"balanced_accuracy": accuracy_score},
                    estimator_class=SVC,
                    hyperparam_space=evolvable_hyperparams,
                    genetic_params={"generations": 2, "population_size": 5}, seed=42,
                    use_parallel=False)
    clf = opt.optimize_clf(5, 2)
    assert clf is not None

def test_validate_hyperparam_space_none():
    x, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="hyperparam_space is None"):
        Optimizer(estimator_class=DecisionTreeClassifier, hyperparam_space=None,
                  genetic_params={},
                  features=x, labels=y)

def test_validate_hyperparam_space_invalid_params():
    x, y = load_iris(return_X_y=True)
    invalid_hyperparams = {
        "invalid_param": Hyperparam("invalid_param", 0, 10, 'int')
    }
    invalid_fixed_hyperparams = {
        "max_depth": 10
    }
    hyperparam_space = HyperparameterSpace(evolvable_hyperparams=invalid_hyperparams,
                                           fixed_hyperparams=invalid_fixed_hyperparams)
    with pytest.raises(ValueError,
                       match="Parameters {'invalid_param'} are not parameters of DecisionTreeClassifier"):
        Optimizer(estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
                  genetic_params={}, features=x, labels=y)

def test_validate_hyperparam_space_valid_params():
    x, y = load_iris(return_X_y=True)
    valid_hyperparams = {
        "max_depth": Hyperparam("max_depth", 1, 10, 'int')
    }
    hyperparam_space = HyperparameterSpace(evolvable_hyperparams=valid_hyperparams,
                                           fixed_hyperparams={})
    optimizer = Optimizer(estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
                          genetic_params={}, features=x, labels=y)
    assert optimizer is not None