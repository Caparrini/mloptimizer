from abc import ABC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from mloptimizer.genoptimizer import Hyperparam, BaseOptimizer


class TreeOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a tree classifier from sklearn.tree.DecisionTreeClassifier.
    It inherits from BaseOptimizer.

    """

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)

        if "scale_pos_weight" in individual_dict.keys():
            class_weight = {0: 1, 1: individual_dict["scale_pos_weight"]}
        else:
            class_weight = "balanced"

        clf = DecisionTreeClassifier(criterion="gini",
                                     class_weight=class_weight,
                                     splitter="best",
                                     max_features=None,
                                     max_depth=individual_dict['max_depth'],
                                     min_samples_split=individual_dict['min_samples_split'],
                                     min_samples_leaf=individual_dict['min_samples_leaf'],
                                     min_impurity_decrease=individual_dict['min_impurity_decrease'],
                                     # min_weight_fraction_leaf=individual_dict['min_weight_fraction_leaf'],
                                     ccp_alpha=individual_dict['ccp_alpha'],
                                     max_leaf_nodes=None,
                                     random_state=None)
        return clf

    @staticmethod
    def get_default_hyperparams():
        default_hyperparams = {
            "min_samples_split": Hyperparam("min_samples_split", 2, 50, int),
            "min_samples_leaf": Hyperparam("min_samples_leaf", 1, 20, int),
            "max_depth": Hyperparam("max_depth", 2, 20, int),
            "min_impurity_decrease": Hyperparam("min_impurity_decrease", 0, 150, float, 1000),
            "ccp_alpha": Hyperparam("ccp_alpha", 0, 300, float, 100000)
        }
        return default_hyperparams


class ForestOptimizer(TreeOptimizer, ABC):
    """
    Class for the optimization of a forest classifier from sklearn.ensemble.RandomForestClassifier.
    It inherits from TreeOptimizer.

    """

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)

        clf = RandomForestClassifier(n_estimators=individual_dict['n_estimators'],
                                     criterion="gini",
                                     max_depth=individual_dict['max_depth'],
                                     max_samples=individual_dict['max_samples'],
                                     min_weight_fraction_leaf=individual_dict['min_weight_fraction_leaf'],
                                     min_impurity_decrease=individual_dict['min_impurity_decrease'],
                                     max_features=individual_dict['max_features'],
                                     max_leaf_nodes=None,
                                     bootstrap=True,
                                     oob_score=True,
                                     n_jobs=-1,
                                     random_state=None,
                                     verbose=0,
                                     warm_start=False,
                                     class_weight="balanced"
                                     )
        return clf

    @staticmethod
    def get_default_hyperparams():
        default_hyperparams = {
            "max_features": Hyperparam("max_features", 1, 100, float, 100),
            "n_estimators": Hyperparam("n_estimators", 5, 250, int),
            "max_samples": Hyperparam("max_samples", 10, 100, float, 100),
            "max_depth": Hyperparam("max_depth", 2, 14, int),
            "min_impurity_decrease": Hyperparam("min_impurity_decrease", 0, 500, float, 100),
            # min_weight_fraction_leaf must be a float in the range [0.0, 0.5]
            "min_weight_fraction_leaf": Hyperparam("min_weight_fraction_leaf", 0, 50, float, 100)
        }
        return default_hyperparams


class ExtraTreesOptimizer(ForestOptimizer, ABC):
    """
    Class for the optimization of a extra trees classifier from sklearn.ensemble.ExtraTreesClassifier.
    It inherits from ForestOptimizer.
    """

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)

        class_weight = "balanced"

        if "scale_pos_weight" in individual_dict.keys():
            perc_class_one = individual_dict["scale_pos_weight"]
            total = 10
            class_one = total * perc_class_one
            class_zero = total - class_one
            real_weight_zero = total / (2 * class_zero)
            real_weight_one = total / (2 * class_one)
            class_weight = {0: real_weight_zero, 1: real_weight_one}

        clf = ExtraTreesClassifier(n_estimators=individual_dict['n_estimators'],
                                   criterion="gini",
                                   max_depth=individual_dict['max_depth'],
                                   # min_samples_split=individual_dict['min_samples_split'],
                                   # min_samples_leaf=individual_dict['min_samples_leaf'],
                                   min_weight_fraction_leaf=individual_dict['min_weight_fraction_leaf'],
                                   min_impurity_decrease=individual_dict['min_impurity_decrease'],
                                   max_features=individual_dict['max_features'],
                                   max_samples=individual_dict['max_samples'],
                                   max_leaf_nodes=None,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=-1,
                                   random_state=None,
                                   verbose=0,
                                   warm_start=False,
                                   class_weight=class_weight
                                   )
        return clf


class GradientBoostingOptimizer(ForestOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from sklearn.ensemble.GradientBoostingClassifier.
    It inherits from ForestOptimizer.
    """

    def get_hyperparams(self):
        """
        Hyperparams for the creation of individuals (relative to the algorithm)
        These hyperparams define the name of the hyperparam, min value, max value, and type

        :return: list of hyperparams
        """
        hyperparams = super(GradientBoostingOptimizer, self).get_hyperparams()
        # learning_rate
        hyperparams["learning_rate"] = Hyperparam('learning_rate', 1, 10000, float, 1000000)
        # subsample
        del hyperparams["max_samples"]
        # subsample must be a float in the range (0.0, 1.0]
        hyperparams["subsample"] = Hyperparam('subsample', 10, 100, float, 100)
        # Return all the hyperparams
        return hyperparams

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = GradientBoostingClassifier(n_estimators=individual_dict['n_estimators'],
                                         criterion="friedman_mse",
                                         max_depth=individual_dict['max_depth'],
                                         # min_samples_split=individual_dict['min_samples_split'],
                                         # min_samples_leaf=individual_dict['min_samples_leaf'],
                                         min_weight_fraction_leaf=individual_dict['min_weight_fraction_leaf'],
                                         min_impurity_decrease=individual_dict['min_impurity_decrease'],
                                         max_features=individual_dict['max_features'],
                                         max_leaf_nodes=None,
                                         random_state=None,
                                         verbose=0,
                                         warm_start=False,
                                         learning_rate=individual_dict['learning_rate'],
                                         subsample=individual_dict['subsample'])
        return clf
