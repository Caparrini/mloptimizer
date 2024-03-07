import os
import random
from abc import ABCMeta, abstractmethod
from random import randint

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import creator, tools, base
from deap.algorithms import varAnd
from sklearn.metrics import accuracy_score

from mloptimizer.evaluation import train_score
from mloptimizer.plots import plotly_logbook, plotly_search_space
from mloptimizer.hyperparams import HyperparameterSpace
from mloptimizer.aux import Tracker, utils


class BaseOptimizer(object):
    """
    Base class for the optimization of a classifier

    Attributes
    ----------
    features : np.array
        np.array with the features
    labels : np.array
        np.array with the labels
    hyperparam_space : HyperparameterSpace
        object with the hyperparameter space: fixed and evolvable hyperparams
    eval_function : func
        function to evaluate the model from X, y, clf
    score_function : func
        function to score from y, y_pred
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
    __metaclass__ = ABCMeta

    def __init__(self, features: np.array, labels: np.array, folder=os.curdir, log_file="mloptimizer.log",
                 hyperparam_space: HyperparameterSpace = None,
                 eval_function=train_score,
                 score_function=accuracy_score, seed=random.randint(0, 1000000),
                 use_parallel=False, use_mlflow=False):
        """
        Creates object BaseOptimizer.

        Parameters
        ----------
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
        eval_function : func, optional (default=kfold_stratified_score)
            function to evaluate the model from X, y, clf
        score_function : func, optional (default=balanced_accuracy_score)
            function to score from y, y_pred
        use_parallel : bool, optional (default=False)
            flag to use parallel processing
        use_mlflow : bool, optional (default=False)
            flag to use mlflow
        seed : int, optional (default=0)
            seed for the random functions (deap, models, and splits on evaluations)
        """
        # Input mandatory variables
        self.features = features
        self.labels = labels
        # Input search space hyperparameters
        self.hyperparam_space = hyperparam_space

        self.eval_function = eval_function
        self.score_function = score_function

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
        self.tracker = Tracker(name="mloptimizer", folder=folder, log_file=log_file, use_mlflow=self.use_mlflow)

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
        [next_subclasses.extend(BaseOptimizer.get_subclasses(x)) for x in subclasses]
        return [*subclasses, *next_subclasses]

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
            ps.append(randint(self.hyperparam_space.evolvable_hyperparams[k].min_value,
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

    @abstractmethod
    def get_clf(self, individual):
        """
        Method to get the classifier from an individual. Abstract method implemented in each specific optimizer.

        Parameters
        ----------
        individual : individual
            individual to convert

        Returns
        -------
        clf : classifier
            classifier specific for the optimizer
        """
        pass

    def evaluate_clf(self, individual):
        """
        Method to evaluate the classifier from an individual. It uses the eval_function to evaluate the classifier.

        Parameters
        ----------
        individual : individual
            individual to convert

        Returns
        -------
        mean : float
            mean of the evaluation
        """
        mean = self.eval_function(self.features, self.labels, self.get_clf(individual),
                                  score_function=self.score_function)
        self.tracker.log_evaluation(self.get_clf(individual), mean)
        return (mean,)

    def population_2_df(self):
        """
        Method to convert the population to a pandas dataframe

        Returns
        -------
        df : pandas dataframe
            dataframe with the population
        """
        data = []
        n = 0
        for p in self.populations:
            for i in p:
                i_hyperparams = self.get_clf(i[0]).get_params()
                i_hyperparams['fitness'] = i[1].values[0]
                i_hyperparams['population'] = n
                data.append(i_hyperparams)
            n += 1

        df = pd.DataFrame(data)
        return df

    def _write_population_file(self, filename=None):
        """
        Method to write the population to a csv file

        Parameters
        ----------
        filename : str, optional (default=None)
            filename to save the population
        """
        if filename is None:
            filename = os.path.join(self.tracker.results_path, 'populations.csv')
        self.population_2_df().sort_values(by=['fitness'], ascending=False
                                           ).to_csv(filename, index=False)

    def _write_logbook_file(self, filename=None):
        """
        Method to write the logbook to a csv file

        Parameters
        ----------
        filename : str, optional (default=None)
            filename to save the logbook
        """
        if filename is None:
            filename = os.path.join(self.tracker.results_path, 'logbook.csv')
        pd.DataFrame(self.logbook).to_csv(filename, index=False)

    def _read_logbook_file(self, filename=None):
        """
        Method to read the logbook from a csv file

        Parameters
        ----------
        filename : str, optional (default=None)
            filename to read the logbook
        """
        if filename is None:
            filename = os.path.join(self.tracker.results_path, 'logbook.csv')
        data = []
        if os.path.exists(filename):
            data = pd.read_csv(filename)
        else:
            self.tracker.optimization_logger.error("File {} does not exist".format(filename))
        return data

    def optimize_clf(self, population: int = 10, generations: int = 3,
                     checkpoint: str = None, opt_run_folder_name: str = None) -> object:
        """
        Method to optimize the classifier. It uses the custom_ea_simple method to optimize the classifier.

        Parameters
        ----------
        population : int, optional (default=10)
            number of individuals in each generation
        generations : int, optional (default=3)
            number of generations
        checkpoint : str, optional (default=None)
            path to the checkpoint file
        opt_run_folder_name : str, optional (default=None)
            name of the folder where the execution will be saved

        Returns
        -------
        clf : classifier
            classifier with the best hyperparams
        """
        # Log initialization
        self.tracker.start_optimization(type(self).__name__)

        # Creation of individual and population
        toolbox = base.Toolbox()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start_gen = 0
        # Using deap, custom for decision tree
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Parallel https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
        # TODO: Scoop compatibility
        if self.use_parallel:
            try:
                import multiprocessing
                pool = multiprocessing.Pool()
                toolbox.register("map", pool.map)
            except ImportError as e:
                # self.optimization_logger.warning("Multiprocessing not available: {}".format(e))
                self.tracker.optimization_logger.warning("Multiprocessing not available: {}".format(e))

        toolbox.register("individual", self.init_individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Tools
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(10)
        self.logbook = tools.Logbook()

        if checkpoint:
            self.tracker.optimization_logger, _ = utils.init_logger(os.path.join(checkpoint, "opt.log"))
            cp = joblib.load(checkpoint)
            self.tracker.optimization_logger.info("Initiating from checkpoint {}...".format(checkpoint))
            pop = cp['population']
            start_gen = cp['generation'] + 1
            hof = cp['halloffame']
            self.logbook = cp['logbook']
            random.setstate(cp['rndstate'])
            # Extract checkpoint_path from checkpoint file
            self.tracker.opt_run_checkpoint_path = os.path.dirname(checkpoint)
            self.tracker.results_path = os.path.join(self.tracker.opt_run_checkpoint_path, "results")
            self.tracker.graphics_path = os.path.join(self.tracker.opt_run_checkpoint_path, "graphics")
        else:

            self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            # Create checkpoint_path from date and algorithm
            self.tracker.start_checkpoint(opt_run_folder_name)

        # Methods for genetic algorithm
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt,
                         low=[x.min_value for x in self.hyperparam_space.evolvable_hyperparams.values()],
                         up=[x.max_value for x in self.hyperparam_space.evolvable_hyperparams.values()],
                         indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("evaluate", self.evaluate_clf)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        fpop, self.logbook, hof = self.custom_ea_simple(pop, toolbox, self.logbook, cxpb=0.5, mutpb=0.5,
                                                        checkpoint_path=self.tracker.opt_run_checkpoint_path,
                                                        start_gen=start_gen, ngen=generations, stats=stats,
                                                        halloffame=hof)

        self.tracker.optimization_logger.info("LOGBOOK: \n{}".format(self.logbook))
        self.tracker.optimization_logger.info("HALL OF FAME: {} individuals".format(len(hof)))

        halloffame_classifiers = list(map(self.get_clf, hof))
        halloffame_fitness = [ind.fitness.values[:] for ind in hof]
        self.tracker.log_clfs(classifiers_list=halloffame_classifiers, generation=-1,
                              fitness_list=halloffame_fitness)

        self._write_population_file()
        self._write_logbook_file()
        # self.plot_logbook(logbook=logbook)
        hyperparam_names = list(self.hyperparam_space.evolvable_hyperparams.keys())
        hyperparam_names.append("fitness")
        population_df = self.population_2_df()
        df = population_df[hyperparam_names]
        g = plotly_search_space(df)
        g.write_html(os.path.join(self.tracker.graphics_path, "search_space.html"))
        plt.close()

        g2 = plotly_logbook(self.logbook, population_df)
        # g2.savefig(os.path.join(self.graphics_path, "logbook.png"))
        g2.write_html(os.path.join(self.tracker.graphics_path, "logbook.html"))
        plt.close()

        #TODO: Log the best model (or top n)

        return self.get_clf(hof[0])

    def custom_ea_simple(self, population, toolbox, logbook,
                         cxpb, mutpb, start_gen=0, ngen=4, checkpoint_path=None, stats=None,
                         halloffame=None, verbose=__debug__, checkpoint_flag=True):
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution

        The algorithm takes in a population and evolves it in place using the
        :meth:`varAnd` method. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evaluations for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varAnd` function. The pseudocode goes as follow ::

            evaluate(population)
            for g in range(ngen):
                population = select(population, len(population))
                offspring = varAnd(population, toolbox, cxpb, mutpb)
                evaluate(offspring)
                population = offspring

        As stated in the pseudocode above, the algorithm goes as follow. First, it
        evaluates the individuals with an invalid fitness. Second, it enters the
        generational loop where the selection procedure is applied to entirely
        replace the parental population. The 1:1 replacement ratio of this
        algorithm **requires** the selection procedure to be stochastic and to
        select multiple times the same individual, for example,
        :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
        Third, it applies the :func:`varAnd` function to produce the next
        generation population. Fourth, it evaluates the new individuals and
        compute the statistics on this population. Finally, when *ngen*
        generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.

        .. note::

            Using a non-stochastic selection method will result in no selection as
            the operator selects *n* individuals from a pool of *n*.

        This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox.

        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
           Basic Algorithms and Operators", 2000.
        """

        if checkpoint_flag and (checkpoint_path is None or not os.path.isdir(checkpoint_path)):
            error_msg = "checkpoint_flag is True and checkpoint_path {} " \
                        "is not a folder or does not exist".format(checkpoint_path)
            self.tracker.optimization_logger.error(error_msg)
            raise NotADirectoryError(error_msg)

        # Begin the generational process

        for gen in range(start_gen, ngen + 1):
            progress_gen_path = os.path.join(self.tracker.progress_path, "Generation_{}.csv".format(gen))
            progress_gen_file = open(progress_gen_path, "w")
            header_progress_gen_file = "i;total;Individual;fitness\n"
            progress_gen_file.write(header_progress_gen_file)
            progress_gen_file.close()
            self.tracker.optimization_logger.info("Generation: {}".format(gen))
            # Vary the pool of individuals
            population = varAnd(population, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            c = 1
            evaluations_pending = len(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                self.tracker.optimization_logger.info(
                    "Fitting individual (informational purpose): gen {} - ind {} of {}".format(
                        gen, c, evaluations_pending
                    )
                )
                ind.fitness.values = fit
                ind_formatted = self.individual2dict(ind)
                progress_gen_file = open(progress_gen_path, "a")
                progress_gen_file.write(
                    "{};{};{};{}\n".format(c,
                                           evaluations_pending,
                                           ind_formatted, fit)
                )
                progress_gen_file.close()
                c = c + 1

            halloffame.update(population)

            record = stats.compile(population) if stats else {}

            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                self.tracker.optimization_logger.info(logbook.stream)

            # Select the next generation individuals
            population = toolbox.select(population, len(population))

            halloffame_classifiers = list(map(self.get_clf, halloffame[:2]))
            halloffame_fitness = [ind.fitness.values[:] for ind in halloffame[:2]]
            self.tracker.log_clfs(classifiers_list=halloffame_classifiers, generation=gen,
                                  fitness_list=halloffame_fitness)

            # Store the space hyperparams and fitness for each individual
            self.populations.append([[ind, ind.fitness] for ind in population])

            if checkpoint_flag:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(population=population, generation=gen, halloffame=halloffame,
                          logbook=logbook, rndstate=random.getstate())

                cp_file = os.path.join(checkpoint_path, "cp_gen_{}.pkl".format(gen))
                joblib.dump(cp, cp_file)
            self._write_population_file()
            self._write_logbook_file()

        return population, logbook, halloffame
