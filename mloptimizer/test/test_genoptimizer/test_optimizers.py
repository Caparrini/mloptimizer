import pytest
from sklearn.datasets import load_iris, load_breast_cancer
import time

from mloptimizer.genoptimizer import TreeOptimizer, ForestOptimizer, ExtraTreesOptimizer, \
    GradientBoostingOptimizer, SVCOptimizer, XGBClassifierOptimizer, KerasClassifierOptimizer, \
    CustomXGBClassifierOptimizer, CatBoostClassifierOptimizer, \
    BaseOptimizer
from mloptimizer.evaluation import kfold_score, train_score, train_test_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, \
    balanced_accuracy_score, precision_score, recall_score, \
    average_precision_score, log_loss, mean_squared_error, mean_absolute_error, \
    r2_score, explained_variance_score, max_error, mean_squared_log_error, \
    median_absolute_error, mean_poisson_deviance, mean_gamma_deviance, \
    mean_tweedie_deviance, mean_absolute_percentage_error


@pytest.mark.parametrize('target_metric',
                         (balanced_accuracy_score, accuracy_score))
@pytest.mark.parametrize('optimizer',
                         (TreeOptimizer, ForestOptimizer,
                          ExtraTreesOptimizer, GradientBoostingOptimizer,
                          XGBClassifierOptimizer,
                          SVCOptimizer,
                          KerasClassifierOptimizer))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer(optimizer, dataset, target_metric):
    X, y = dataset(return_X_y=True)
    opt = optimizer(X, y, score_function=target_metric)
    clf = opt.optimize_clf(2, 2)
    assert clf is not None


@pytest.mark.parametrize('target_metric',
                         (balanced_accuracy_score, accuracy_score))
@pytest.mark.parametrize('optimizer',
                         (TreeOptimizer, ForestOptimizer,
                          ExtraTreesOptimizer, GradientBoostingOptimizer,
                          # XGBClassifierOptimizer, Cannot be used with parallel
                          SVCOptimizer
                                 # , KerasClassifierOptimizer, cannot be used with parallel
                          ))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer_use_parallel(optimizer, dataset, target_metric):
    X, y = dataset(return_X_y=True)
    my_seed = 25
    population = 50
    generations = 4

    opt_with_parallel = optimizer(X, y, score_function=target_metric, seed=my_seed, use_parallel=True)

    start_time_parallel = time.time()
    clf_with_parallel = opt_with_parallel.optimize_clf(population, generations)
    end_time_parallel = time.time()

    opt = optimizer(X, y, score_function=target_metric, seed=my_seed, use_parallel=False)

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


def test_get_subclasses():
    subclasses = BaseOptimizer.get_subclasses(BaseOptimizer)
    subclasses_names = [
        'TreeOptimizer', 'ForestOptimizer', 'ExtraTreesOptimizer',
        'GradientBoostingOptimizer', 'SVCOptimizer', 'XGBClassifierOptimizer',
        'CustomXGBClassifierOptimizer', 'KerasClassifierOptimizer', 'CatBoostClassifierOptimizer',
        'SklearnOptimizer'
    ]
    assert all([subclass.__name__ in subclasses_names for subclass in subclasses]) and \
           len(subclasses) == len(subclasses_names)


@pytest.mark.parametrize('optimizer',
                         (TreeOptimizer, ForestOptimizer,
                          ExtraTreesOptimizer, GradientBoostingOptimizer,
                          XGBClassifierOptimizer,
                                 # SVCOptimizer,KerasClassifierOptimizer
                          ))
@pytest.mark.parametrize('target_metric', (balanced_accuracy_score, accuracy_score))
@pytest.mark.parametrize('target_score', (kfold_score, train_score, train_test_score))
def test_reproducibility(optimizer, target_metric, target_score):
    X, y = load_iris(return_X_y=True)
    population = 2
    generations = 2
    seed = 25
    distinct_seed = 2
    optimizer1 = optimizer(X, y, score_function=target_metric,
                           eval_function=target_score, seed=seed)
    result1 = optimizer1.optimize_clf(population=population, generations=generations)
    optimizer2 = optimizer(X, y, score_function=target_metric,
                           eval_function=target_score, seed=seed)
    result2 = optimizer2.optimize_clf(population=population, generations=generations)
    optimizer3 = optimizer(X, y, score_function=target_metric,
                           eval_function=target_score, seed=distinct_seed)
    result3 = optimizer3.optimize_clf(population=population, generations=generations)
    assert str(result1) == str(result2)
    assert str(result1) != str(result3)
