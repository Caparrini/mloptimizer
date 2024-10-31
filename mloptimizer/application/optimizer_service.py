import random
import numpy as np
from mloptimizer.domain.hyperspace import HyperparameterSpace
from mloptimizer.domain.evaluation import train_score
from mloptimizer.domain.optimization import Optimizer
from mloptimizer.application.utils import get_default_fitness_score


class OptimizerService:
    """
    Service layer class that orchestrates the optimization process using the genetic algorithm.
    This class interacts with the Optimizer and provides a high-level interface for the API.
    """

    def __init__(self, estimator_class, hyperparam_space: HyperparameterSpace,
                 genetic_params: dict,
                 eval_function: callable = train_score, scoring = None,
                 metrics:dict = None, seed: int=None, use_parallel: bool=False):
        """
        Initialize the OptimizerService.

        Parameters:
        ----------
        estimator_class : class
            The machine learning model class (e.g., RandomForestClassifier, SVC).
        hyperparam_space : HyperparameterSpace
            The hyperparameter space defining which parameters to evolve and optimize.
        genetic_params : dict
            The parameters for the genetic algorithm optimization.
        eval_function : callable, optional
            The function to evaluate the model performance, defaulting to train_score.
        metrics : dict, optional
            The metrics to use to evaluate the performance of the classifier.
        seed : int, optional
            Random seed for reproducibility, defaults to a random value.
        use_parallel : bool, optional
            Whether to use parallel processing for optimization, defaults to False.
        """
        self.estimator_class = estimator_class
        self.hyperparam_space = hyperparam_space
        self.genetic_params = genetic_params
        self.eval_function = eval_function or train_score
        self.scoring = get_default_fitness_score(estimator_class, scoring)
        self.metrics = metrics
        self.seed = seed or random.randint(0, 1000000)
        self.use_parallel = use_parallel
        self.optimizer = None

    def optimize(self, features: np.array, labels: np.array):
        """
        Optimize the machine learning model using genetic algorithms.

        Parameters:
        ----------
        features : np.array
            Feature set for the optimization process.
        labels : np.array
            Label set for the optimization process.
        generations : int, optional (default=10)
            Number of generations for the genetic algorithm to run.
        population_size : int, optional (default=50)
            Size of the population for the genetic algorithm.

        Returns:
        --------
        Optimized model: The trained model with the best-found hyperparameters.
        """
        # Create a new instance of the Optimizer for this run
        self.optimizer = Optimizer(
            estimator_class=self.estimator_class,
            features=features,
            labels=labels,
            hyperparam_space=self.hyperparam_space,
            eval_function=self.eval_function,
            fitness_score=self.scoring,
            metrics=self.metrics,
            seed=self.seed,
            use_parallel=self.use_parallel
        )

        # Run the genetic algorithm-based optimization and return the best model
        best_model = self.optimizer.optimize_clf(**self.genetic_params)
        return best_model

    def set_hyperparameter_space(self, hyperparam_space: HyperparameterSpace):
        """
        Allows the user to update the hyperparameter space dynamically.

        Parameters:
        ----------
        hyperparam_space : HyperparameterSpace
            The new hyperparameter space to be used in optimization.
        """
        self.hyperparam_space = hyperparam_space

    def set_eval_function(self, eval_function: callable):
        """
        Allows the user to set or change the evaluation function dynamically.

        Parameters:
        ----------
        eval_function : callable
            A new evaluation function to use during the optimization.
        """
        self.eval_function = eval_function

    def set_genetic_params(self, genetic_params: dict):
        """
        Allows the user to set or change the genetic algorithm parameters dynamically.

        Parameters:
        ----------
        genetic_params : dict
            The new genetic algorithm parameters to use during the optimization.
        """
        for key, value in genetic_params.items():
            self.genetic_params[key] = value
