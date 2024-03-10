import random
from deap import creator, base, tools
import numpy as np
from mloptimizer.hyperparams import HyperparameterSpace


class DeapOptimizer:
    def __init__(self, hyperparam_space: HyperparameterSpace = None, use_parallel=False,  seed=None):
        """
        Class to start the parameters for the use of DEAP library.

        Parameters
        ----------
        hyperparam_space : HyperparameterSpace
            hyperparameter space
        use_parallel : bool
            flag to use parallel processing
        seed : int
            seed for the random functions

        Attributes
        ----------
        hyperparam_space : HyperparameterSpace
            hyperparameter space
        use_parallel : bool
            flag to use parallel processing
        seed : int
            seed for the random functions
        toolbox : deap.base.Toolbox
            toolbox for the optimization
        eval_dict : dict
            dictionary with the evaluation of the individuals
        logbook : list
            list of logbook
        stats : deap.tools.Statistics
            statistics of the optimization
        """
        self.hyperparam_space = hyperparam_space
        self.use_parallel = use_parallel
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.toolbox = base.Toolbox()
        self.eval_dict = {}
        self.logbook = None
        self.stats = None
        self.setup()

    def init_individual(self, pcls):
        """
        Method to create an individual

        Parameters
        ----------
        pcls : class
            class of the individual

        Returns
        -------
        ind : individual
            individual
        """
        ps = []
        for k in self.hyperparam_space.evolvable_hyperparams.keys():
            ps.append(random.randint(self.hyperparam_space.evolvable_hyperparams[k].min_value,
                                     self.hyperparam_space.evolvable_hyperparams[k].max_value)
                      )
        individual_initialized = pcls(ps)
        return individual_initialized

    def individual2dict(self, individual):
        """
        Method to convert an individual to a dictionary of hyperparams

        Parameters
        ----------
        individual : individual
            individual to convert

        Returns
        -------
        individual_dict : dict
            dictionary of hyperparams
        """
        individual_dict = {}
        keys = list(self.hyperparam_space.evolvable_hyperparams.keys())
        for i in range(len(keys)):
            individual_dict[keys[i]] = self.hyperparam_space.evolvable_hyperparams[keys[i]].correct(individual[i])
        return {**individual_dict, **self.hyperparam_space.fixed_hyperparams}

    def setup(self):
        """
        Method to set the parameters for the optimization.
        """
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        start_gen = 0
        # Using deap, custom for decision tree
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Parallel https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
        if self.use_parallel:
            try:
                #from scoop import futures
                import multiprocessing
                pool = multiprocessing.Pool()
                self.toolbox.register("map", pool.map)
            except ImportError as e:
                # self.optimization_logger.warning("Multiprocessing not available: {}".format(e))
                # self.tracker.optimization_logger.warning("Multiprocessing not available: {}".format(e))
                print("Multiprocessing not available: {}".format(e))

        self.toolbox.register("individual", self.init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
