import os
import random
import numpy as np
from sklearn.base import is_regressor, is_classifier

from mloptimizer.domain.evaluation import train_score
from mloptimizer.domain.hyperspace import HyperparameterSpace
from mloptimizer.infrastructure.tracking import Tracker
from mloptimizer.domain.evaluation import Evaluator



class Optimizer:
    """
    Base class for the optimization of a classifier

    Attributes
    ----------
    estimator_class : class
        class of the classifier
    features : np.array
        np.array with the features
    labels : np.array
        np.array with the labels
    hyperparam_space : HyperparameterSpace
        object with the hyperparameter space: fixed and evolvable hyperparams
    genetic_params : dict
        dictionary with the parameters of the genetic algorithm
    evaluator : Evaluator
        object to evaluate the classifier
    eval_dict : dict
        dictionary with the evaluation of the individuals
    populations : list
        list of populations
    logbook : list
        list of logbook
    seed : int
        seed for the random functions
    use_parallel : bool
        flag to use parallel processing
    use_mlflow : bool
        flag to use mlflow
    """

    def __init__(self, estimator_class, features: np.array, labels: np.array,
                 folder=os.curdir, log_file="mloptimizer.log",
                 hyperparam_space: HyperparameterSpace = None,
                 genetics_params: dict = None,
                 eval_function=train_score,
                 fitness_score=None, metrics=None, seed=random.randint(0, 1000000),
                 use_parallel=False, use_mlflow=False):
        """
        Creates object BaseOptimizer.

        Parameters
        ----------
        estimator_class : class
            class of the classifier
        features : np.array
            np.array with the features
        labels : np.array
            np.array with the labels
        folder : path, optional (default=os.curdir)
            folder to store the structure of files and folders product of executions
        log_file : str, optional (default="mloptimizer.log")
            log file name
        hyperparam_space : HyperparameterSpace, optional (default=None)
            object with the hyperparameter space: fixed and evolvable hyperparams
        genetics_params : dict, optional (default=None)
            dictionary with the parameters of the genetic algorithm
        eval_function : func, optional (default=train_score)
            function to evaluate the model from X, y, clf
        fitness_score : str, optional (default="accuracy")
            fitness score to use to evaluate the performance of the classifier
        use_parallel : bool, optional (default=False)
            flag to use parallel processing
        use_mlflow : bool, optional (default=False)
            flag to use mlflow
        seed : int, optional (default=0)
            seed for the random functions (deap, models, and splits on evaluations)
        """
        from mloptimizer.domain.population import IndividualUtils
        # Model class
        self.estimator_class = estimator_class
        if not is_classifier(self.estimator_class) and not is_regressor(self.estimator_class):
            raise ValueError(f"The estimator class {self.estimator_class} must be a classifier or a regressor")

        # Input mandatory variables
        self.features = features
        self.labels = labels
        # Input search space hyperparameters
        self.hyperparam_space = hyperparam_space
        self._validate_hyperparam_space()

        # ML Evaluator
        # if metrics is None:
        #     metrics = {"accuracy": accuracy_score}
        self.genetic_params = genetics_params

        # State vars
        self.eval_dict = {}
        self.populations = []
        self.logbook = None
        self.mlopt_seed = None
        self.set_mlopt_seed(seed)

        # Parallel
        self.use_parallel = use_parallel

        # mlflow
        self.use_mlflow = use_mlflow

        # Tracker
        self.tracker = Tracker(name="mloptimizer", folder=folder, log_file=log_file, use_mlflow=self.use_mlflow,
                               use_parallel=self.use_parallel)

        # Evaluator
        self.individual_utils = IndividualUtils(hyperparam_space=self.hyperparam_space,
                                                estimator_class=self.estimator_class, mlopt_seed=self.mlopt_seed)
        self.evaluator = Evaluator(estimator_class=self.estimator_class, features=features, labels=labels,
                                   eval_function=eval_function, fitness_score=fitness_score,
                                   metrics=metrics, tracker=self.tracker,
                                   individual_utils=self.individual_utils)

        # DeapOptimizer
        # self.deap_optimizer = None
        self.genetic_algorithm = None
        #self.runs = []

    def _validate_hyperparam_space(self):
        """
        Method to validate the hyperparam_space
        """
        if self.hyperparam_space is None:
            raise ValueError("hyperparam_space is None")
        # Default params of the estimator_class
        default_estimator_params = set(self.estimator_class().get_params().keys())
        hyperparam_space_params = set(self.hyperparam_space.get_all_params().keys())
        illegal_params = hyperparam_space_params - default_estimator_params

        if len(illegal_params) > 0:
            raise ValueError(f"Parameters {illegal_params} are not parameters of {self.estimator_class.__name__}")

        # Check values of the parameters

    def set_mlopt_seed(self, seed):
        """
        Method to set the seed for the random functions

        Parameters
        ----------
        seed : int
            seed for the random functions
        """
        self.mlopt_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_subclasses(my_class):
        """
        Method to get all the subclasses of a class
        (in this case use to get all the classifiers that can be optimized).

        Parameters
        ----------
        my_class : class
            class to get the subclasses

        Returns
        -------
        list
            list of subclasses
        """
        subclasses = my_class.__subclasses__()
        if len(subclasses) == 0:
            return []
        next_subclasses = []
        [next_subclasses.extend(Optimizer.get_subclasses(x)) for x in subclasses]
        return [*subclasses, *next_subclasses]

    def get_clf(self, individual):
        # individual_dict = self.deap_optimizer.individual2dict(individual)
        # clf = self.estimator_class(random_state=self.mlopt_seed, **individual_dict)
        return self.individual_utils.get_clf(individual)

    #def optimize_clf(self, population_size: int = 10, generations: int = 3,
    #                 cxpb=0.5, mutpb=0.5, tournsize=4, indpb=0.5, n_elites=10,
    #                 checkpoint: str = None, opt_run_folder_name: str = None) -> object:
    def optimize_clf(self, population_size: int = 10, generations: int = 3,
                     cxpb=0.5, mutpb=0.5, tournsize=4, indpb=0.5, n_elites=10,
                     checkpoint: str = None, opt_run_folder_name: str = None) -> object:
        """
        Method to optimize the classifier. It uses the custom_ea_simple method to optimize the classifier.

        Parameters
        ----------
        population_size : int, optional (default=10)
            number of individuals in each generation
        generations : int, optional (default=3)
            number of generations
        cxpb : float, optional (default=0.5)
            crossover probability
        mutpb : float, optional (default=0.5)
            mutation probability
        tournsize : int, optional (default=4)
            number of individuals to select in the tournament
        indpb : float, optional (default=0.5)
            independent probability for each attribute to be mutated
        n_elites : int, optional (default=10)
            number of elites to keep in the next generation
        checkpoint : str, optional (default=None)
            path to the checkpoint file
        opt_run_folder_name : str, optional (default=None)
            name of the folder where the execution will be saved

        Returns
        -------
        clf : classifier
            classifier with the best hyperparams
        """
        #from mloptimizer.domain.optimization import DeapOptimizer, GeneticAlgorithmRunner
        from mloptimizer.domain.optimization import GeneticAlgorithm
        # Log initialization
        self.tracker.start_optimization(type(self).__name__, generations=generations)

        # Creation of folders and checkpoint
        self.tracker.start_checkpoint(opt_run_folder_name, estimator_class=self.estimator_class)

        # Creation of deap optimizer
        #self.deap_optimizer = DeapOptimizer(hyperparam_space=self.hyperparam_space, seed=self.mlopt_seed,
        #                                    use_parallel=self.use_parallel,
        #                                    maximize=is_classifier(self.estimator_class))
        # Creation of genetic algorithm runner
        #ga_runner = GeneticAlgorithmRunner(deap_optimizer=self.deap_optimizer,
        #                                   tracker=self.tracker,
        #                                   seed=self.mlopt_seed,
        #                                   evaluator=self.evaluator)
        self.genetic_algorithm = GeneticAlgorithm(hyperparam_space=self.hyperparam_space,
                                                  tracker=self.tracker,
                                                  seed=self.mlopt_seed,
                                                  evaluator=self.evaluator,
                                                  use_parallel=self.use_parallel,
                                                  maximize=is_classifier(self.estimator_class))


        # Run genetic algorithm
        population, logbook, hof = self.genetic_algorithm.custom_run(population_size=population_size,
                                                                     n_generations=generations,
                                                                     cxpb=cxpb, mutation_prob=mutpb,
                                                                     n_elites=n_elites,
                                                                     tournsize=tournsize,
                                                                     indpb=indpb,
                                                                     checkpoint=checkpoint)

        #self.runs.append(ga_runner)
        #self.logbook = logbook

        # self.populations = population
        # self.populations.append([[ind, ind.fitness] for ind in population])

        # Log and save results
        # self._log_and_save_results(hof)

        return self.individual_utils.get_clf(hof[0])
