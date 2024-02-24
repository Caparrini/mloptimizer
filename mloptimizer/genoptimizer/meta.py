from mloptimizer.genoptimizer import BaseOptimizer


class SklearnOptimizer(BaseOptimizer):
    """
    This class is a wrapper for scikit-learn classifiers. It is used to optimize hyperparameters for scikit-learn
    classifiers using genetic algorithms. The class inherits from the BaseOptimizer class and implements the
    get_clf and get_default_hyperparams methods. The get_clf method returns a scikit-learn classifier with the
    hyperparameters specified in the individual. The get_default_hyperparams method returns a dictionary with the
    default hyperparameters for the scikit-learn classifier.
    """
    def __init__(self, clf_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_class = clf_class

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = self.clf_class(random_state=self.mlopt_seed, **individual_dict)
        return clf

    def get_default_hyperparams(self):
        """
        This method returns a dictionary with the default hyperparameters for the scikit-learn classifier.
        TODO: Implement this method based on the clf_class attribute.
        """
        return {}
