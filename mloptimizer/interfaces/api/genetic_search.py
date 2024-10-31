from mloptimizer.application import OptimizerService, HyperparameterSpaceService
import random


class GeneticSearch:
    """
    Genetic algorithm-based optimization for hyperparameter tuning.

    The `GeneticOptimizer` provides an interface for optimizing an estimator's hyperparameters
    using a genetic algorithm. It supports cross-validation and parallel computation.

    Parameters
    ----------
    estimator_class : class
        The class of the estimator to be optimized.

    hyperparam_space : dict or HyperparameterSpace
        The hyperparameter search space as a dictionary or a `HyperparameterSpace` object.

    genetic_params_dict : dict, optional (default=None)
        Genetic algorithm parameters for the optimization process. If None, default parameters are used.

    eval_function : callable, optional (default=None)
        Custom evaluation function for the estimator. If None, the default estimator's score method is used.

    seed : int, optional (default=None)
        Random seed for reproducibility. If None, a random seed is generated.

    scoring : str or callable, optional (default=None)
        Scoring method to evaluate the estimator's performance. If None, the estimator’s default score method is used.

    use_parallel : bool, optional (default=False)
        Whether to run the optimization in parallel. If True, parallel processing is enabled.

    cv : int, cross-validation generator or an iterable, optional (default=None)
        Cross-validation splitting strategy. If None, the default cross-validation strategy is used.

    Attributes
    ----------
    best_estimator_ : estimator
        The estimator with the best found hyperparameters after fitting.

    best_params_ : dict
        The hyperparameters that produced the best performance during the optimization.

    cv_results_ : list of dicts
        A log of the optimization progress, containing details such as fitness scores and hyperparameters
        evaluated during each generation.
    """
    default_genetic_params = {
        "generations": 20,
        "population_size": 15,
        'cxpb':  0.5, 'mutpb': 0.5,
        'n_elites': 10, 'tournsize': 3, 'indpb': 0.05
    }

    def __init__(self, estimator_class, hyperparam_space, eval_function=None,
                 genetic_params_dict=None, seed=None, scoring=None, use_parallel=False,
                 cv=None):
        """Initialize the GeneticOptimizer with the necessary components."""
        # Set the genetic algorithm parameters
        self.genetic_params = self.default_genetic_params
        self.set_genetic_params(**(genetic_params_dict or {}))

        self.optimizer_service = OptimizerService(
            estimator_class=estimator_class,
            hyperparam_space=hyperparam_space,
            genetic_params=self.genetic_params,
            eval_function=eval_function,
            scoring=scoring,
            seed=seed or random.randint(0, 1000000),
            use_parallel=use_parallel
        )

        self.hyperparam_service = HyperparameterSpaceService()
        self.cv = cv  # Optional cross-validator

        # Attributes to store the best results
        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.logbook_ = None
        self.populations_ = None


    def fit(self, X, y):
        """
        Run the genetic algorithm optimization to fit the best model.

        Parameters
        ----------
        X : np.array
            Feature set for the optimization process.

        y : np.array
            Label set for the optimization process.

        Returns
        -------
        self : object
            Fitted `GeneticOptimizer` object.
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("Features and labels must not be empty.")

        # Perform optimization via the optimizer service
        estimator_with_best_params = self.optimizer_service.optimize(X, y)
        self.best_estimator_ = estimator_with_best_params.fit(X, y)

        # Extract best hyperparameters from the optimizer service
        self.best_params_ = self.best_estimator_.get_params()

        # Store the detailed cross-validation or genetic algorithm results TODO
        self.cv_results_ = self.optimizer_service.optimizer.genetic_algorithm.logbook

        # Store logbook
        self.logbook_ = self.optimizer_service.optimizer.genetic_algorithm.logbook

        # Store population df
        self.populations_ = self.optimizer_service.optimizer.genetic_algorithm.population_2_df()

        return self

    def predict(self, X):
        """
        Make predictions using the best estimator found by the optimization process.

        Parameters
        ----------
        X : np.array
            Input features to predict labels.

        Returns
        -------
        y_pred : np.array
            Predicted labels.
        """
        if self.best_estimator_ is None:
            raise ValueError("The model must be fitted before predicting.")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """
        Return the score of the best estimator on the given test data and labels.

        Parameters
        ----------
        X : np.array
            Test feature set.

        y : np.array
            True labels for scoring.

        Returns
        -------
        score : float
            Score of the best estimator on the test data.
        """
        if self.best_estimator_ is None:
            raise ValueError("The model must be fitted before scoring.")
        return self.best_estimator_.score(X, y)

    def set_hyperparameter_space(self, hyperparam_space):
        """
        Set or update the hyperparameter space for the optimization process.

        Parameters
        ----------
        hyperparam_space : HyperparameterSpace
            The hyperparameter space object to be used for optimization.
        """
        self.optimizer_service.set_hyperparameter_space(hyperparam_space)

    def get_evolvable_hyperparams(self):
        """
        Get the evolvable hyperparameters from the hyperparameter space.

        Returns
        -------
        evolvable_hyperparams : dict
            Dictionary of evolvable hyperparameters.
        """
        return self.optimizer_service.hyperparam_space.evolvable_hyperparams

    def set_eval_function(self, eval_function: callable):
        """
        Set or update the evaluator function for the optimization process.

        Parameters
        ----------
        eval_function : callable
            A new evaluation function for the optimization process.
        """
        self.optimizer_service.set_eval_function(eval_function)

    def load_default_hyperparameter_space(self, estimator_class):
        """
        Load a default hyperparameter space for the given estimator using the HyperparameterSpaceService.

        Parameters
        ----------
        estimator_class : class
            The estimator class for which to load the default hyperparameter space.

        Returns
        -------
        HyperparameterSpace
            The loaded hyperparameter space object.
        """
        return self.hyperparam_service.load_default_hyperparameter_space(estimator_class)

    def load_hyperparameter_space(self, file_path):
        """
        Load a hyperparameter space from a file using the HyperparameterSpaceService.

        Parameters
        ----------
        file_path : str
            The path to the file containing the hyperparameter space.

        Returns
        -------
        HyperparameterSpace
            The loaded hyperparameter space object.
        """
        return self.hyperparam_service.load_hyperparameter_space(file_path)

    def save_hyperparameter_space(self, file_path, overwrite=False):
        """
        Save the current hyperparameter space to a file using the HyperparameterSpaceService.

        Parameters
        ----------
        file_path : str
            The path to the file where the hyperparameter space will be saved.
        overwrite : bool, optional (default=False)
            Whether to overwrite the existing file if it exists.
        """
        if self.optimizer_service.hyperparam_space is None:
            raise ValueError("No hyperparameter space is set for saving.")
        self.hyperparam_service.save_hyperparameter_space(
            self.optimizer_service.hyperparam_space, file_path, overwrite)

    def get_params(self):
        """
        Get parameters for this optimizer.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "estimator_class": self.optimizer_service.estimator_class,
            "hyperparam_space": self.optimizer_service.hyperparam_space,
            "eval_function": self.optimizer_service.eval_function,
            "seed": self.optimizer_service.seed,
            "use_parallel": self.optimizer_service.use_parallel,
            "cv": self.cv,
        }

    def set_params(self, **params):
        """
        Set the parameters of this optimizer.

        Parameters
        ----------
        **params : dict
            Estimator parameters to update.

        Returns
        -------
        self : object
            Updated `GeneticOptimizer` object.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_genetic_params(self):
        """
        Get the genetic algorithm parameters.

        Returns
        -------
        genetic_params : dict
            Genetic algorithm parameters.
        """
        return self.genetic_params

    def set_genetic_params(self, **params):
        """
        Set the genetic algorithm parameters.

        Parameters
        ----------
        **params : dict
            Genetic algorithm parameters to update.

        Returns
        -------
        self : object
            Updated `GeneticOptimizer` object.
        """
        for param, value in params.items():
            self.genetic_params[param] = value
        return self
