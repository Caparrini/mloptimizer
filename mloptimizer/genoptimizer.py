import os
import random
import shutil
from abc import ABCMeta, abstractmethod, ABC
from datetime import datetime
from random import randint

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from deap import creator, tools, base
from deap.algorithms import varAnd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mloptimizer import miscellaneous
from mloptimizer.alg_wrapper import CustomXGBClassifier, generate_model
from mloptimizer.model_evaluation import kfold_stratified_score
from mloptimizer.plots import plotly_logbook, plotly_search_space


class Param(object):
    """
    Class to define a param to optimize. It defines the name, min value, max value and type.
    This is used to control the precision of the param and avoid multiple evaluations
    with close values of the param due to decimal positions.


    Attributes
    ----------
    name : str
        Name of the param. It will be used as key in a dictionary
    min_value : int
        Minimum value of the param
    max_value : int
        Maximum value of the param
    type : type
        Type of the param (int, float, 'nexp', 'x10')
    denominator : int, optional (default=100)
        Optional param in case the type=float
    values_str : list, optional (default=[])
        List of string with possible values (TODO)
    """

    def __init__(self, name: str, min_value: int, max_value: int, param_type,
                 denominator: int = 100, values_str: list = None):
        """
        Creates object Param.

        Parameters
        ----------
        name : str
            Name of the param. It will be used as key in a dictionary
        min_value : int
            Minimum value of the param
        max_value : int
            Maximum value of the param
        type : type
            Type of the param (int, float, 'nexp', 'x10')
        denominator : int, optional (default=100)
            Optional param in case the type=float
        values_str : list, optional (default=[])
            List of string with possible values (TODO)
        """
        if values_str is None:
            values_str = []
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.type = param_type
        self.denominator = denominator
        self.values_str = values_str

    def correct(self, value: int):
        """
        Returns the real value of the param in case some mutation could surpass the limits.
            1) Verifies the input is int
            2) Enforce min and max value
            3) Apply the type of value

        Parameters
        ----------
        value : int
            Value to correct

        Returns
        -------
        ret : int, float
            Corrected value
        """
        # Input value must be int
        value = int(value)
        ret = None
        # Verify the value is in range
        if value > self.max_value:
            value = self.max_value
        elif value < self.min_value:
            value = self.min_value
        # Apply the type of value
        if self.type == int:
            ret = value
        elif self.type == float:
            ret = float(value) / self.denominator
            # ret = round(value, self.decimals)
        elif self.type == "nexp":
            ret = 10 ** (-value)
        elif self.type == "x10":
            ret = value * 10
        return ret

    def __eq__(self, other_param):
        """Overrides the default implementation"""
        equals = (self.name == other_param.name and self.min_value == other_param.min_value and
                  self.type == other_param.type and self.denominator == other_param.denominator and
                  self.max_value == other_param.max_value)
        return equals

    def __str__(self):
        """Overrides the default implementation"""
        if type(self.type) == type:
            type_str = self.type.__name__
        elif type(self.type) == str:
            type_str = "'{}'".format(self.type)

        if self.type == float:
            param_str = "Param('{}', {}, {}, {}, {})".format(
                self.name,
                self.min_value,
                self.max_value,
                type_str,
                self.denominator
            )
        else:
            param_str = "Param('{}', {}, {}, {})".format(
                self.name,
                self.min_value,
                self.max_value,
                type_str
            )

        return param_str

    def __repr__(self):
        """Overrides the default implementation"""
        return self.__str__()


class BaseOptimizer(object):
    """
    Base class for the optimization of a classifier

    Attributes
    ----------
    features : np.array
        np.array with the features
    labels : np.array
        np.array with the labels
    custom_params : dict
        dictionary with custom params
    custom_fixed_params : dict
        dictionary with custom fixed params
    fixed_params : dict
        dictionary with fixed params
    params : dict
        dictionary with params
    folder : path
        folder to store the structure of files and folders product of executions
    log_file : str
        log file name
    mloptimizer_logger : logger
        logger for the mloptimizer
    optimization_logger : logger
        logger for the optimization
    eval_function : func
        function to evaluate the model from X, y, clf
    score_function : func
        function to score from y, y_pred
    exe_path : path
        path to the folder where the execution will be saved
    checkpoint_path : path
        path to the folder where the checkpoints will be saved
    progress_path : path
        path to the folder where the progress will be saved
    results_path : path
        path to the folder where the results will be saved
    graphics_path : path
        path to the folder where the graphics will be saved
    eval_dict : dict
        dictionary with the evaluation of the individuals
    populations : list
        list of populations
    logbook : list
        list of logbook
    seed : int
        seed for the random functions TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self, features: np.array, labels: np.array, folder=None, log_file="mloptimizer.log",
                 custom_params: dict = {},
                 custom_fixed_params: dict = {}, eval_function=kfold_stratified_score,
                 score_function=balanced_accuracy_score, seed=0):
        """
        Creates object BaseOptimizer.

        Parameters
        ----------
        features : np.array
            np.array with the features
        labels : np.array
            np.array with the labels
        folder : path, optional (default=None)
            folder to store the structure of files and folders product of executions
        log_file : str, optional (default="mloptimizer.log")
            log file name
        custom_params : dict, optional (default={})
            dictionary with custom params
        custom_fixed_params : dict, optional (default={})
            dictionary with custom fixed params
        eval_function : func, optional (default=kfold_stratified_score)
            function to evaluate the model from X, y, clf
        score_function : func, optional (default=balanced_accuracy_score)
            function to score from y, y_pred
        seed : int, optional (default=0)
            seed for the random functions TODO
        """
        # Input mandatory variables
        self.features = features
        self.labels = labels
        # Input parameters (optional)
        self.custom_params = custom_params
        self.custom_fixed_params = custom_fixed_params
        self.fixed_params = self.get_fixed_params()
        self.params = self.get_params()
        # Main folder (autogenerated if None)
        self.folder = miscellaneous.create_optimization_folder(folder)
        # Log files
        self.mloptimizer_logger, self.log_file = miscellaneous.init_logger(log_file, self.folder)
        self.optimization_logger = None
        self.eval_function = eval_function
        self.score_function = score_function
        # Paths
        self.exe_path = None
        self.checkpoint_path = None
        self.progress_path = None
        self.progress_path = None
        self.results_path = None
        self.graphics_path = None
        # State vars
        self.eval_dict = {}
        self.populations = []
        self.logbook = None
        self.seed = seed

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

    def get_folder(self):
        """
        Method to get the folder where the execution will be saved
        """
        return self.folder

    def get_log_file(self):
        """
        Method to get the log file name
        """
        return self.log_file

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
        for k in self.params.keys():
            ps.append(randint(self.params[k].min_value, self.params[k].max_value))
        ind = pcls(ps)
        return ind

    @abstractmethod
    def individual2dict(self, individual):
        """
        Method to convert an individual to a dictionary of params

        Parameters
        ----------
        individual : individual
            individual to convert

        Returns
        -------
        individual_dict : dict
            dictionary of params
        """
        individual_dict = {}
        keys = list(self.params.keys())
        for i in range(len(keys)):
            individual_dict[keys[i]] = self.params[keys[i]].correct(individual[i])
        return {**individual_dict, **self.get_fixed_params()}

    @abstractmethod
    def get_params(self):
        """
        Method to get the params to optimize. First the fixed params are removed from the list, then
        the custom override the default params.

        Returns
        -------
        params : dict
            dictionary of params
        """
        params = {}
        default_params = self.get_default_params()
        for k in self.custom_fixed_params.keys():
            default_params.pop(k, None)

        for k in default_params.keys():
            if k in self.custom_params:
                params[k] = self.custom_params[k]
            else:
                params[k] = default_params[k]

        # Return all the params
        return params

    @abstractmethod
    def get_fixed_params(self):
        """
        Method to get the fixed params dictionary. These params are stores using
        only the name of the param and the target values (not as Param objects that are only used
        in hyperparameters that are evolved).

        Returns
        -------
        fixed_params : dict
            dictionary of fixed params
        """
        fixed_params = {**self.get_default_fixed_params(), **self.custom_fixed_params}
        return fixed_params

    @abstractmethod
    def get_default_fixed_params(self):
        """
        Method to get the default fixed params dictionary. Empty by default.

        Returns
        -------
        default_fixed_params : dict
            dictionary of default fixed params
        """
        default_fixed_params = {

        }
        return default_fixed_params

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
                i_params = self.get_clf(i[0]).get_params()
                i_params['fitness'] = i[1].values[0]
                i_params['population'] = n
                data.append(i_params)
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
            filename = os.path.join(self.results_path, 'populations.csv')
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
            filename = os.path.join(self.results_path, 'logbook.csv')
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
            filename = os.path.join(self.results_path, 'logbook.csv')
        data = []
        if os.path.exists(filename):
            data = pd.read_csv(filename)
        else:
            self.optimization_logger.error("File {} does not exist".format(filename))
        return data

    def optimize_clf(self, population: int = 10, generations: int = 3,
                     checkpoint: str = None, exe_folder: str = None) -> object:
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
        exe_folder : str, optional (default=None)
            name of the folder where the execution will be saved

        Returns
        -------
        clf : classifier
            classifier with the best params
        """
        self.mloptimizer_logger.info("Initiating genetic optimization...")
        self.mloptimizer_logger.info("Algorithm: {}".format(type(self).__name__))
        # Creation of individual and population
        toolbox = base.Toolbox()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start_gen = 0
        # self.file_out.write("Optimizing accuracy:\n")
        # Using deap, custom for decision tree
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Paralel
        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)

        toolbox.register("individual", self.init_individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Tools
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(10)
        self.logbook = tools.Logbook()

        if checkpoint:
            self.optimization_logger, _ = miscellaneous.init_logger(os.path.join(checkpoint, "opt.log"))
            cp = joblib.load(checkpoint)
            self.optimization_logger.info("Initiating from checkpoint {}...".format(checkpoint))
            pop = cp['population']
            start_gen = cp['generation'] + 1
            hof = cp['halloffame']
            self.logbook = cp['logbook']
            random.setstate(cp['rndstate'])
            # Extract checkpoint_path from checkpoint file
            self.checkpoint_path = os.path.dirname(checkpoint)
            self.results_path = os.path.join(self.checkpoint_path, "results")
            self.graphics_path = os.path.join(self.checkpoint_path, "graphics")
        else:

            self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            # Create checkpoint_path from date and algorithm
            if exe_folder:
                exe_name = exe_folder
            else:
                exe_name = "{}_{}".format(
                    datetime.now().strftime("%Y%m%d_%H%M%S"),
                    type(self).__name__)
            self.exe_path = os.path.join(self.folder, exe_name)
            self.checkpoint_path = os.path.join(self.exe_path, "checkpoints")
            self.results_path = os.path.join(self.exe_path, "results")
            self.graphics_path = os.path.join(self.exe_path, "graphics")
            self.progress_path = os.path.join(self.exe_path, "progress")
            if os.path.exists(self.exe_path):
                shutil.rmtree(self.exe_path)
            os.mkdir(self.exe_path)
            os.mkdir(self.checkpoint_path)
            os.mkdir(self.results_path)
            os.mkdir(self.graphics_path)
            os.mkdir(self.progress_path)
            self.optimization_logger, _ = miscellaneous.init_logger(os.path.join(self.exe_path, "opt.log"))

        # Methods for genetic algorithm
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[x.min_value for x in self.params.values()],
                         up=[x.max_value for x in self.params.values()], indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("evaluate", self.evaluate_clf)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        fpop, self.logbook, hof = self.custom_ea_simple(pop, toolbox, self.logbook, cxpb=0.5, mutpb=0.5,
                                                        checkpoint_path=self.checkpoint_path,
                                                        start_gen=start_gen, ngen=generations, stats=stats,
                                                        halloffame=hof)

        self.optimization_logger.info("LOGBOOK: \n{}".format(self.logbook))
        self.optimization_logger.info("HALL OF FAME: {} individuals".format(len(hof)))

        for i in range(len(hof)):
            best_score = hof[i].fitness.values[:]
            self.optimization_logger.info("Individual TOP {}".format(i + 1))
            self.optimization_logger.info("Individual accuracy: {}".format(best_score))
            self.optimization_logger.info("Best classifier: {}".format(str(self.get_clf(hof[i]))))
            self.optimization_logger.info("Params: {}".format(str(self.get_clf(hof[i]).get_params())))

        # self.file_out.write("LOGBOOK: \n"+str(logbook)+"\n")
        # self.file_out.write("Best accuracy: "+str(best_score[0])+"\n")
        # self.file_out.write("Best classifier(without parameter formating(DECIMALS)): "+str(self.get_clf(hof[0])))
        self._write_population_file()
        self._write_logbook_file()
        # self.plot_logbook(logbook=logbook)
        param_names = list(self.get_params().keys())
        param_names.append("fitness")
        population_df = self.population_2_df()
        df = population_df[param_names]
        g = plotly_search_space(df)
        g.write_html(os.path.join(self.graphics_path, "search_space.html"))
        plt.close()

        g2 = plotly_logbook(self.logbook, population_df)
        #g2.savefig(os.path.join(self.graphics_path, "logbook.png"))
        g2.write_html(os.path.join(self.graphics_path, "logbook.html"))
        plt.close()

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
            self.optimization_logger.error(error_msg)
            raise NotADirectoryError(error_msg)

        # Begin the generational process

        for gen in range(start_gen, ngen + 1):
            progress_gen_path = os.path.join(self.progress_path, "Generation_{}.csv".format(gen))
            progress_gen_file = open(progress_gen_path, "w")
            header_progress_gen_file = "i;total;Individual;fitness\n"
            progress_gen_file.write(header_progress_gen_file)
            progress_gen_file.close()
            self.optimization_logger.info("Generation: {}".format(gen))
            # Vary the pool of individuals
            population = varAnd(population, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            c = 1
            evaluations_pending = len(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                self.optimization_logger.info(
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
                self.optimization_logger.info(logbook.stream)

            # Select the next generation individuals
            population = toolbox.select(population, len(population))

            for i in range(len(halloffame[:2])):
                best_score = halloffame[i].fitness.values[:]
                self.optimization_logger.info("Individual TOP {}".format(i + 1))
                self.optimization_logger.info("Individual accuracy: {}".format(best_score))
                self.optimization_logger.info("Best classifier: {}".format(str(self.get_clf(halloffame[i]))))
                self.optimization_logger.info(
                    "Params: {}".format(str(self.get_clf(halloffame[i]).get_params())))

            # Store the space param and fitness for each
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
    def get_default_params():
        default_params = {
            "min_samples_split": Param("min_samples_split", 2, 50, int),
            "min_samples_leaf": Param("min_samples_leaf", 1, 20, int),
            "max_depth": Param("max_depth", 2, 50, int),
            "min_impurity_decrease": Param("min_impurity_decrease", 0, 150, float, 1000),
            "ccp_alpha": Param("ccp_alpha", 0, 300, float, 100000)
        }
        return default_params


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
    def get_default_params():
        default_params = {
            "max_features": Param("max_features", 1, 100, float, 100),
            "n_estimators": Param("n_estimators", 5, 250, int),
            "max_samples": Param("max_samples", 10, 100, float, 100),
            "max_depth": Param("max_depth", 2, 14, int),
            "min_impurity_decrease": Param("min_impurity_decrease", 0, 500, float, 100),
            # min_weight_fraction_leaf must be a float in the range [0.0, 0.5]
            "min_weight_fraction_leaf": Param("min_weight_fraction_leaf", 0, 50, float, 100)
        }
        return default_params


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
    def get_params(self):
        """
        Params for the creation of individuals (relative to the algorithm)
        These params define the name of the param, min value, max value, and type

        :return: list of params
        """
        params = super(GradientBoostingOptimizer, self).get_params()
        # learning_rate
        params["learning_rate"] = Param('learning_rate', 1, 10000, float, 1000000)
        # subsample
        del params["max_samples"]
        # subsample must be a float in the range (0.0, 1.0]
        params["subsample"] = Param('subsample', 10, 100, float, 100)
        # Return all the params
        return params

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


class XGBClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from xgboost.XGBClassifier.
    It inherits from BaseOptimizer.
    """
    @staticmethod
    def get_default_params():
        default_params = {
            'colsample_bytree': Param("colsample_bytree", 3, 10, float, 10),
            'gamma': Param("gamma", 0, 20, int),
            'learning_rate': Param("learning_rate", 1, 100, float, 1000),
            'max_depth': Param("max_depth", 3, 30, int),
            'n_estimators': Param("n_estimators", 100, 500, int),
            'subsample': Param("subsample", 700, 1000, float, 1000),
            'scale_pos_weight': Param("scale_pos_weight", 15, 40, float, 100)
        }
        return default_params

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = xgb.XGBClassifier(base_score=0.5,
                                booster='gbtree',
                                colsample_bytree=individual_dict['colsample_bytree'],
                                colsample_bylevel=1,
                                eval_metric='logloss',
                                gamma=individual_dict['gamma'],
                                learning_rate=individual_dict['learning_rate'],
                                max_depth=individual_dict['max_depth'],
                                n_estimators=individual_dict['n_estimators'],
                                n_jobs=-1,
                                objective='binary:logistic',
                                random_state=0,
                                # reg_alpha=0,
                                # reg_lambda=1,
                                scale_pos_weight=individual_dict['scale_pos_weight'],
                                seed=0,
                                subsample=individual_dict['subsample'],
                                # tree_method="gpu_hist"
                                )
        return clf


class CustomXGBClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from alg_wrapper.CustomXGBClassifier.
    It inherits from BaseOptimizer.
    """
    @staticmethod
    def get_default_params():
        default_params = {
            'eta': Param("eta", 0, 100, float, 100),
            'colsample_bytree': Param("colsample_bytree", 3, 10, float, 10),
            'alpha': Param("alpha", 0, 100, float, 100),
            'lambda': Param("lambda", 0, 100, float, 100),
            'gamma': Param("gamma", 0, 100, float, 100),
            'max_depth': Param("max_depth", 3, 14, int),
            'subsample': Param("subsample", 70, 100, float, 100),
            'num_boost_round': Param("num_boost_round", 2, 100, int),
            'scale_pos_weight': Param("scale_pos_weight", 10, 10000, float, 100),
            'min_child_weight': Param("min_child_weight", 0, 100, float, 10)
        }
        return default_params

    def get_default_fixed_params(self):
        default_fixed_params = {
            'obj': None,
            'feval': None
        }
        return default_fixed_params

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = CustomXGBClassifier(base_score=0.5,
                                  booster="gbtree",
                                  eval_metric="auc",
                                  eta=individual_dict['eta'],
                                  gamma=individual_dict['gamma'],
                                  subsample=individual_dict['subsample'],
                                  colsample_bylevel=1,
                                  colsample_bytree=individual_dict['colsample_bytree'],
                                  max_delta_step=0,
                                  max_depth=individual_dict['max_depth'],
                                  min_child_weight=individual_dict['min_child_weight'],
                                  seed=1,
                                  alpha=individual_dict['alpha'],
                                  reg_lambda=individual_dict['lambda'],
                                  num_boost_round=individual_dict['num_boost_round'],
                                  scale_pos_weight=individual_dict['scale_pos_weight'],
                                  obj=self.fixed_params['obj'],
                                  feval=self.fixed_params['feval'])
        return clf


class CatBoostClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from catboost.CatBoostClassifier.
    It inherits from BaseOptimizer.
    """
    @staticmethod
    def get_default_params():
        default_params = {
            'eta': Param("eta", 1, 10, float, 10),
            'max_depth': Param("max_depth", 3, 16, int),  # Max is 16
            'n_estimators': Param("n_estimators", 100, 500, int),
            'subsample': Param("subsample", 700, 1000, float, 1000),
        }
        return default_params

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = CatBoostClassifier(
            **individual_dict, auto_class_weights="Balanced",
            bootstrap_type='Bernoulli'
        )
        return clf


class KerasClassifierOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a gradient boosting classifier from keras.wrappers.scikit_learn.KerasClassifier.
    It inherits from BaseOptimizer.
    """
    @staticmethod
    def get_default_params():
        default_params = {
            'epochs': Param("epochs", 1, 10, "x10"),
            'batch_size': Param("batch_size", 1, 5, "x10"),
            'learning_rate': Param("learning_rate", 1, 20, float, 1000),
            'layer_1': Param("layer_1", 10, 50, "x10"),
            'layer_2': Param("layer_2", 5, 20, "x10"),
            'dropout_rate_1': Param("dropout_rate_1", 0, 5, float, 10),
            'dropout_rate_2': Param("dropout_rate_2", 0, 5, float, 10),
        }
        return default_params

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        print(individual_dict)
        clf = KerasClassifier(build_fn=generate_model,
                              **individual_dict)
        return clf


class SVCOptimizer(BaseOptimizer, ABC):
    """
    Class for the optimization of a support vector machine classifier from sklearn.svm.SVC.
    It inherits from BaseOptimizer.
    """
    @staticmethod
    def get_default_params():
        default_params = {
            'C': Param("C", 1, 10000, float, 10),
            'degree': Param("degree", 0, 6, int),
            'gamma': Param("gamma", 10, 100000000, float, 100)
        }
        return default_params

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = SVC(C=individual_dict['C'],
                  cache_size=8000000,
                  class_weight="balanced",
                  coef0=0.0,
                  decision_function_shape='ovr',
                  degree=individual_dict['degree'], gamma=individual_dict['gamma'],
                  kernel='rbf',
                  max_iter=100000,
                  probability=False,
                  random_state=None,
                  shrinking=True,
                  tol=0.001,
                  verbose=False
                  )
        return clf
