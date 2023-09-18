import pytest
from sklearn.datasets import load_iris, load_breast_cancer

from mloptimizer.genoptimizer import TreeOptimizer, ForestOptimizer, ExtraTreesOptimizer, \
    GradientBoostingOptimizer, SVCOptimizer, XGBClassifierOptimizer, KerasClassifierOptimizer, \
    CustomXGBClassifierOptimizer, CatBoostClassifierOptimizer, \
    BaseOptimizer
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
                          #ExtraTreesOptimizer, GradientBoostingOptimizer,
                          XGBClassifierOptimizer,
                          #SVCOptimizer,
                          KerasClassifierOptimizer))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer(optimizer, dataset, target_metric):
    X, y = dataset(return_X_y=True)
    opt = optimizer(X, y, score_function=target_metric)
    clf = opt.optimize_clf(2, 1)
    assert clf is not None


def test_get_subclasses():
    subclasses = BaseOptimizer.get_subclasses(BaseOptimizer)
    subclasses_names = [
        'TreeOptimizer', 'ForestOptimizer', 'ExtraTreesOptimizer',
        'GradientBoostingOptimizer', 'SVCOptimizer', 'XGBClassifierOptimizer',
        'CustomXGBClassifierOptimizer', 'KerasClassifierOptimizer', 'CatBoostClassifierOptimizer'
    ]
    assert all([subclass.__name__ in subclasses_names for subclass in subclasses]) and \
           len(subclasses) == len(subclasses_names)

