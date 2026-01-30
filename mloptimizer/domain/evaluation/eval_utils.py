import time
import numpy as np
from sklearn.base import clone
from mloptimizer.domain.evaluation.model_evaluation import score_metrics


class EvaluationResult:
    """
    Extended evaluation result that includes per-fold CV data for sklearn compatibility.

    Attributes
    ----------
    metrics : dict
        Aggregated metrics (mean across folds)
    fold_metrics : list of dict
        Per-fold metrics for each CV fold
    fit_times : list of float
        Fit time for each fold
    score_times : list of float
        Score time for each fold
    n_splits : int
        Number of CV splits
    cv_strategy : str
        Name of the CV strategy used
    """

    def __init__(self, metrics, fold_metrics=None, fit_times=None, score_times=None,
                 n_splits=None, cv_strategy=None):
        self.metrics = metrics  # Aggregated (mean) metrics
        self.fold_metrics = fold_metrics or []  # Per-fold metrics
        self.fit_times = fit_times or []
        self.score_times = score_times or []
        self.n_splits = n_splits
        self.cv_strategy = cv_strategy

    def get_mean_metrics(self):
        """Return aggregated metrics dict (for backwards compatibility)."""
        return self.metrics

    def get_std_metrics(self):
        """Return std of metrics across folds."""
        if not self.fold_metrics:
            return {}
        return {
            k: np.std([fold[k] for fold in self.fold_metrics])
            for k in self.fold_metrics[0].keys()
        }

    def get_fold_score(self, fold_idx, metric_name):
        """Return score for a specific fold and metric."""
        if fold_idx < len(self.fold_metrics):
            return self.fold_metrics[fold_idx].get(metric_name)
        return None

    def get_mean_fit_time(self):
        """Return mean fit time across folds."""
        return np.mean(self.fit_times) if self.fit_times else 0.0

    def get_mean_score_time(self):
        """Return mean score time across folds."""
        return np.mean(self.score_times) if self.score_times else 0.0

    def to_sklearn_format(self, param_prefix='param_', params=None):
        """
        Convert to sklearn cv_results_ compatible format for a single trial.

        Returns dict with keys like:
        - mean_test_{metric}
        - std_test_{metric}
        - split{i}_test_{metric}
        - mean_fit_time
        - mean_score_time
        - param_{name} (if params provided)
        """
        result = {}

        # Add parameters if provided
        if params:
            for k, v in params.items():
                result[f'{param_prefix}{k}'] = v

        # Add mean and std for each metric
        for metric_name, mean_value in self.metrics.items():
            result[f'mean_test_{metric_name}'] = mean_value

        std_metrics = self.get_std_metrics()
        for metric_name, std_value in std_metrics.items():
            result[f'std_test_{metric_name}'] = std_value

        # Add per-fold scores
        for fold_idx, fold_metrics in enumerate(self.fold_metrics):
            for metric_name, value in fold_metrics.items():
                result[f'split{fold_idx}_test_{metric_name}'] = value

        # Add timing
        result['mean_fit_time'] = self.get_mean_fit_time()
        result['std_fit_time'] = np.std(self.fit_times) if self.fit_times else 0.0
        result['mean_score_time'] = self.get_mean_score_time()
        result['std_score_time'] = np.std(self.score_times) if self.score_times else 0.0

        return result


def make_crossval_eval(cv, return_extended=True):
    """
    Create a cross-validation evaluation function.

    Parameters
    ----------
    cv : sklearn cross-validator
        Cross-validation strategy
    return_extended : bool, default=False
        If True, returns EvaluationResult with per-fold data.
        If False, returns dict with mean metrics (backwards compatible).

    Returns
    -------
    callable
        Evaluation function that can be used by the optimizer
    """
    # Get CV strategy name
    cv_strategy = type(cv).__name__
    n_splits = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else None

    def eval_fn(estimator, X, y, metrics: dict):
        metric_results = {k: [] for k in metrics}
        fold_metrics_list = []
        fit_times = []
        score_times = []

        for train_idx, test_idx in cv.split(X, y):
            model = clone(estimator)

            # Time the fit
            fit_start = time.time()
            model.fit(X[train_idx], y[train_idx])
            fit_time = time.time() - fit_start
            fit_times.append(fit_time)

            # Time the scoring
            score_start = time.time()
            preds = model.predict(X[test_idx])
            fold_metrics = score_metrics(y[test_idx], preds, metrics)
            score_time = time.time() - score_start
            score_times.append(score_time)

            fold_metrics_list.append(fold_metrics)
            for k in metric_results:
                metric_results[k].append(fold_metrics[k])

        # Calculate aggregated metrics
        aggregated = {k: np.mean(metric_results[k]) for k in metric_results}

        if return_extended:
            return EvaluationResult(
                metrics=aggregated,
                fold_metrics=fold_metrics_list,
                fit_times=fit_times,
                score_times=score_times,
                n_splits=n_splits,
                cv_strategy=cv_strategy
            )
        else:
            # Backwards compatible: return dict only
            return aggregated

    # Store CV metadata on the function for later retrieval
    eval_fn.cv_strategy = cv_strategy
    eval_fn.n_splits = n_splits

    return eval_fn
