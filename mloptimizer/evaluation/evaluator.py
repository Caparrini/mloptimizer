from sklearn.metrics import accuracy_score, balanced_accuracy_score


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
    eval_function : function
        The evaluation function to use to evaluate the performance of the classifier
    fitness_score : str
        The fitness score to use to evaluate the performance of the classifier
    metrics : dict
        The metrics to use to evaluate the performance of the classifier
        Dictionary of the form {"metric_name": metric_function}
    """
    def __init__(self, eval_function, fitness_score="balanced_accuracy", metrics=None):
        if metrics is None:
            self.metrics = _default_metrics()
        else:
            self.metrics = metrics
        self.eval_function = eval_function
        self.fitness_score = fitness_score

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
