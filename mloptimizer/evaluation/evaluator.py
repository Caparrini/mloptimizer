from sklearn.metrics import accuracy_score, balanced_accuracy_score
from mloptimizer.aux import Tracker
import numpy as np


def _default_metrics():
    return {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }


class Evaluator:
    """
    Evaluator class to evaluate the performance of a classifier

    Parameters
    ----------
    features : array-like
        The features to use to evaluate the classifier
    labels : array-like
        The labels to use to evaluate the classifier
    eval_function : function
        The evaluation function to use to evaluate the performance of the classifier
    fitness_score : str
        The fitness score to use to evaluate the performance of the classifier
    metrics : dict
        The metrics to use to evaluate the performance of the classifier
        Dictionary of the form {"metric_name": metric_function}
    tracker : Tracker
        The tracker to use to log the evaluations
    individual_utils : IndividualUtils
        The individual utils to use to get the classifier from the individual
    """

    def __init__(self, features: np.array, labels: np.array, eval_function, fitness_score="accuracy", metrics=None,
                 tracker: Tracker = None, individual_utils=None):
        if metrics is None:
            self.metrics = _default_metrics()
        else:
            self.metrics = metrics
        self.eval_function = eval_function
        self.fitness_score = fitness_score
        self.tracker = tracker
        self.features = features
        self.labels = labels
        self.individual_utils = individual_utils

    def evaluate(self, clf, features, labels):
        """
        Evaluate the performance of a classifier

        Parameters
        ----------
        clf : object
            The classifier to evaluate
        features : array-like
            The features to use to evaluate the classifier
        labels : array-like
            The labels to use to evaluate the classifier

        Returns
        -------
        metrics : dict
            Dictionary of the form {"metric_name": metric_value}
        """
        metrics = self.eval_function(features, labels, clf, self.metrics)
        return metrics

    def evaluate_individual(self, individual):
        clf = self.individual_utils.get_clf(individual)
        metrics = self.evaluate(clf=clf, features=self.features, labels=self.labels)
        self.tracker.log_evaluation(clf, metrics)
        return (metrics[self.fitness_score],)
