import pytest
from mloptimizer.core import Optimizer
from mloptimizer.hyperparams import Hyperparam, HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from xgboost import XGBClassifier
from sklearn.svm import SVC
from mloptimizer.evaluation import kfold_score, train_score, train_test_score
import time
import os
from sklearn.metrics import accuracy_score

custom_evolvable_hyperparams = {
    "min_samples_split": Hyperparam("min_samples_split", 2, 50, 'int'),
    "min_samples_leaf": Hyperparam("min_samples_leaf", 1, 20, 'int'),
    "max_depth": Hyperparam("max_depth", 2, 20, 'int'),
    "min_impurity_decrease": Hyperparam("min_impurity_decrease", 0,
                                        150, 'float', 1000),
    "ccp_alpha": Hyperparam("ccp_alpha", 0, 300, 'float', 100000)
}


@pytest.fixture
def default_metrics_dict():
    return {
        "accuracy": accuracy_score,
    }


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, SVC))
def test_sklearn_optimizer(estimator_class):
    X, y = load_iris(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    mlopt = Optimizer(estimator_class=estimator_class,
                      hyperparam_space=evolvable_hyperparams,
                      features=X, labels=y)
    mlopt.optimize_clf(5, 5)
    assert mlopt is not None


@pytest.mark.parametrize('use_mlflow', [True, False])
def test_mloptimizer(use_mlflow):
    X, y = load_breast_cancer(return_X_y=True)
    mlopt = Optimizer(estimator_class=XGBClassifier,
                      hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier),
                      features=X, labels=y, use_mlflow=use_mlflow)
    mlopt.optimize_clf(5, 5)
    assert mlopt is not None


def test_checkpoints():
    X, y = load_breast_cancer(return_X_y=True)
    mlopt = Optimizer(estimator_class=XGBClassifier,
                      hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier),
                      features=X, labels=y)
    clf = mlopt.optimize_clf(5, 5)

    mlopt2 = Optimizer(estimator_class=XGBClassifier,
                       hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier),
                       features=X, labels=y,
                       seed=mlopt.mlopt_seed)

    checkpoint = os.path.join(mlopt.tracker.opt_run_checkpoint_path,
                              os.listdir(mlopt.tracker.opt_run_checkpoint_path)[-2]
                              )
    clf2 = mlopt2.optimize_clf(5, 5,
                               checkpoint=checkpoint)
    assert mlopt is not None
    assert str(clf) == str(clf2)


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, SVC))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer(estimator_class, dataset, default_metrics_dict):
    X, y = dataset(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    opt = Optimizer(features=X, labels=y, fitness_score="accuracy",
                    metrics=default_metrics_dict, estimator_class=estimator_class,
                    hyperparam_space=evolvable_hyperparams)
    clf = opt.optimize_clf(2, 2)
    assert clf is not None


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, SVC))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer_use_parallel(estimator_class, dataset, default_metrics_dict):
    X, y = dataset(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    my_seed = 25
    population = 50
    generations = 4

    opt_with_parallel = Optimizer(features=X, labels=y, fitness_score="accuracy",
                                  metrics=default_metrics_dict,
                                  seed=my_seed, use_parallel=True,
                                  hyperparam_space=evolvable_hyperparams, estimator_class=estimator_class)

    start_time_parallel = time.time()
    clf_with_parallel = opt_with_parallel.optimize_clf(population, generations)
    end_time_parallel = time.time()

    opt = Optimizer(features=X, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                    seed=my_seed, use_parallel=False,
                    hyperparam_space=evolvable_hyperparams, estimator_class=estimator_class)

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
    assert elapsed_time_parallel < elapsed_time


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, SVC))
@pytest.mark.parametrize('target_score', (kfold_score, train_score, train_test_score))
def test_reproducibility(estimator_class, target_score, default_metrics_dict):
    X, y = load_iris(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    population = 2
    generations = 2
    seed = 25
    distinct_seed = 2
    optimizer1 = Optimizer(features=X, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=seed, estimator_class=estimator_class,
                           hyperparam_space=evolvable_hyperparams)
    result1 = optimizer1.optimize_clf(population_size=population, generations=generations)
    optimizer2 = Optimizer(features=X, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=seed, estimator_class=estimator_class,
                           hyperparam_space=evolvable_hyperparams)
    result2 = optimizer2.optimize_clf(population_size=population, generations=generations)
    optimizer3 = Optimizer(features=X, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=distinct_seed, estimator_class=estimator_class,
                           hyperparam_space=evolvable_hyperparams)
    result3 = optimizer3.optimize_clf(population_size=population, generations=generations)
    assert str(result1) == str(result2)
    assert str(result1) != str(result3)
