import deap.base
from deap.algorithms import eaSimple, varAnd
from deap import tools
from matplotlib import pyplot as plt

from mloptimizer.aux.plots import plotly_search_space, plotly_logbook
from mloptimizer.genetic import DeapOptimizer
from mloptimizer.aux import Tracker
import os
import joblib
import pandas as pd


class GeneticAlgorithmRunner:
    def __init__(self, deap_optimizer: DeapOptimizer, tracker: Tracker,
                 seed, evaluator):
        """
        Class to run the genetic algorithm

        Parameters
        ----------
        deap_optimizer : ~mloptimizer.genetic.DeapOptimizer
            optimizer
        evaluate_individual : function
            function to evaluate the individual
        tracker : ~mloptimizer.aux.Tracker
            tracker
        seed : int
            seed for the random functions

        Attributes
        ----------
        populations : list
            list of populations
        tracker : ~mloptimizer.aux.Tracker
            tracker
        deap_optimizer : ~mloptimizer.genetic.DeapOptimizer
            optimizer
        toolbox : ~deap.base.Toolbox
            toolbox
        seed : int
            seed for the random functions
        """
        self.populations = []
        self.tracker = tracker
        self.deap_optimizer = deap_optimizer
        self.toolbox = self.deap_optimizer.toolbox

        self.evaluator = evaluator
        self.toolbox.register("evaluate", self.evaluator.evaluate_individual)
        self.seed = seed

    def simple_run(self, population_size: int, n_generations: int, cxpb: float = 0.5, mutation_prob: float = 0.5,
                   n_elites: int = 10, tournsize: int = 3, indpb: float = 0.05):
        """
        Method to run the genetic algorithm. This uses the deap eaSimple method.
        It cannot be used to track what happens in each generation.

        Parameters
        ----------
        population_size : int
            size of the population
        n_generations : int
            number of generations
        cxpb : float
            crossover probability
        mutation_prob : float
            mutation probability
        n_elites : int
            number of elites
        tournsize : int
            size of the tournament
        indpb : float
            probability of a gene to be mutated

        Returns
        -------
        population : list
            final population
        logbook : ~deap.tools.Logbook
            logbook
        hof : ~deap.tools.HallOfFame
            hall of fame
        """
        hof, pop = self._pre_run(indpb=indpb, n_elites=n_elites, population_size=population_size,
                                 tournsize=tournsize)
        population, logbook = eaSimple(population=pop, toolbox=self.toolbox, cxpb=cxpb, mutpb=mutation_prob,
                                       ngen=n_generations, stats=self.deap_optimizer.stats, halloffame=hof,
                                       verbose=True)

        return population, logbook, hof

    def run(self, population_size: int, n_generations: int, cxpb: float = 0.5, mutation_prob: float = 0.5,
            n_elites: int = 10, tournsize: int = 3, indpb: float = 0.05, checkpoint: str = None) -> object:
        """
        Method to run the genetic algorithm. This uses the custom_ea_simple method.
        It allows to track what happens in each generation.

        Parameters
        ----------
        population_size : int
            size of the population
        n_generations : int
            number of generations
        cxpb : float
            crossover probability
        mutation_prob : float
            mutation probability
        n_elites : int
            number of elites
        tournsize : int
            size of the tournament
        indpb : float
            probability of a gene to be mutated
        checkpoint : str
            path to the checkpoint file

        Returns
        -------
        population : list
            final population
        logbook : ~deap.tools.Logbook
            logbook
        hof : ~deap.tools.HallOfFame
            hall of fame
        """
        hof, pop = self._pre_run(indpb=indpb, n_elites=n_elites, population_size=population_size,
                                 tournsize=tournsize)

        population, logbook, hof = self.custom_ea_simple(population=pop, toolbox=self.toolbox, cxpb=cxpb,
                                                         mutpb=mutation_prob,
                                                         ngen=n_generations, halloffame=hof, verbose=True,
                                                         checkpoint_path=self.tracker.opt_run_checkpoint_path,
                                                         stats=self.deap_optimizer.stats)

        hyperparam_names = list(self.deap_optimizer.hyperparam_space.evolvable_hyperparams.keys())
        hyperparam_names.append("fitness")
        population_df = self.population_2_df()
        df = population_df[hyperparam_names]
        g = plotly_search_space(df)
        g.write_html(os.path.join(self.tracker.graphics_path, "search_space.html"))
        plt.close()

        g2 = plotly_logbook(logbook, population_df)
        g2.write_html(os.path.join(self.tracker.graphics_path, "logbook.html"))
        plt.close()

        return population, logbook, hof

    def _pre_run(self, indpb: float = 0.5, n_elites: int = 10,
                 population_size: int = 10, tournsize: int = 4):
        """
        Method to initialize the population and the hall of fame

        Parameters
        ----------
        indpb : float
            probability og a gene to be mutated
        n_elites : int
            number of elites
        population_size : int
            size of the population
        tournsize : int
            size of the tournament

        Returns
        -------
        hof : ~deap.tools.HallOfFame
            hall of fame
        pop : list
            population
        """
        # Initialize population
        pop = self.toolbox.population(n=population_size)
        # Initialize hall of fame
        hof = tools.HallOfFame(n_elites)
        # Methods for genetic algorithm
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate", tools.mutUniformInt,
            low=[x.min_value for x in self.deap_optimizer.hyperparam_space.evolvable_hyperparams.values()],
            up=[x.max_value for x in self.deap_optimizer.hyperparam_space.evolvable_hyperparams.values()],
            indpb=indpb
        )
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        # History
        hist = tools.History()
        self.toolbox.decorate("mate", hist.decorator)
        self.toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)
        return hof, pop

    def custom_ea_simple(self, population: list, toolbox: deap.base.Toolbox,
                         cxpb: float = 0.5, mutpb: float = 0.5, start_gen: int = 0, ngen: int = 4,
                         checkpoint_path: str = None,
                         stats: deap.tools.Statistics = None,
                         halloffame: deap.tools.HallOfFame = None, verbose: bool = True,
                         checkpoint_flag: bool = True):
        """
        This algorithm reproduces the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        The code is close to the ~deap.algorithms.eaSimple method, but it has been modified to track
        the progress of the optimization and to save the population and the logbook in each generation.
        More info can be found
        `on deap documentation <https://deap.readthedocs.io/en/master/_modules/deap/algorithms.html#eaSimple>`__

        Parameters
        ----------
        population : list
            A list of individuals.
        toolbox : ~dea.base.Toolbox
            A `toolbox` that contains the evolution operators.
        cxpb : float
            The probability of mating two individuals.
        mutpb : float
            The probability of mutating an individual.
        start_gen : int
            The starting generation number. Used in case of checkpoint.
        ngen : int
            The number of generations.
        checkpoint_path : str
            The path to the checkpoint file.
        stats : ~deap.tools.Statistics
            A `~deap.tools.Statistics` object that is updated inplace, optional.
        halloffame : ~deap.tools.HallOfFame
            A `~deap.tools.HallOfFame` object that contains the best individuals, optional.
        verbose : bool
            Whether or not to log the statistics.
        checkpoint_flag : bool
            Whether or not to save the checkpoint.

        Returns
        -------
        population : list
            The final population.
        logbook : ~deap.tools.Logbook
            A logbook containing the statistics of the evolution.
        halloffame : ~deap.tools.HallOfFame
            A hall of fame object that contains the best individuals.

        References
        --------
        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
              Basic Algorithms and Operators", 2000.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Verify if the checkpoint path exists if checkpoint_flag is True
        if checkpoint_flag and (checkpoint_path is None or not os.path.isdir(checkpoint_path)):
            error_msg = "checkpoint_flag is True and checkpoint_path {} " \
                        "is not a folder or does not exist".format(checkpoint_path)
            self.tracker.optimization_logger.error(error_msg)
            raise NotADirectoryError(error_msg)

        # Begin the generational process
        # import multiprocessing
        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)
        for gen in range(start_gen, ngen + 1):
            self.tracker.start_progress_file(gen)

            # Vary the pool of individuals
            population = varAnd(population, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            c = 1
            evaluations_pending = len(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind_formatted = self.deap_optimizer.individual2dict(ind)
                self.tracker.append_progress_file(gen, c, evaluations_pending, ind_formatted, fit)

                c = c + 1

            halloffame.update(population)

            record = stats.compile(population) if stats else {}

            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                self.tracker.optimization_logger.info(logbook.stream)

            # Select the next generation individuals
            population = toolbox.select(population, len(population))

            # halloffame_classifiers = list(map(self.get_clf, halloffame[:2]))
            # halloffame_fitness = [ind.fitness.values[:] for ind in halloffame[:2]]
            # self.tracker.log_clfs(classifiers_list=halloffame_classifiers, generation=gen,
            #                      fitness_list=halloffame_fitness)
            # Store the space hyperparams and fitness for each individual
            self.populations.append([[ind, ind.fitness] for ind in population])

            if checkpoint_flag:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(population=population, generation=gen, halloffame=halloffame,
                          logbook=logbook, rndstate=self.seed)

                cp_file = os.path.join(checkpoint_path, "cp_gen_{}.pkl".format(gen))
                joblib.dump(cp, cp_file)
            self.tracker.write_population_file(self.population_2_df())
            self.tracker.write_logbook_file(logbook)

        return population, logbook, halloffame

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
                i_hyperparams = self.deap_optimizer.individual2dict(i[0])
                i_hyperparams['fitness'] = i[1].values[0]
                i_hyperparams['population'] = n
                data.append(i_hyperparams)
            n += 1

        df = pd.DataFrame(data)
        return df
