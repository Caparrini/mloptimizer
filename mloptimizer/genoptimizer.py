import multiprocessing
from abc import ABCMeta, abstractmethod, ABC
from random import randint
import numpy as np
import copy
from deap import creator, tools, base, algorithms
from deap.algorithms import varAnd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mloptimizer.model_evaluation import KFoldStratifiedAccuracy, TemporalKFoldAccuracy
from mloptimizer import miscellaneous
from mloptimizer.alg_wrapper import CustomXGBClassifier, generate_model
import xgboost as xgb
from keras.wrappers.scikit_learn import KerasClassifier


class Param(object):
    """
    Object to store param info, type and range of values
    """

    def __init__(self, name, min_value, max_value, param_type, denominator=100, values_str=None):
        """
        Init object

        :param name: (str) Name of the param. It will be use as key in a dictionary
        :param min_value: (int) Minimum value of the param
        :param max_value: (int) Maximum value of the param
        :param param_type: (type) type of the param (int, float, 'nexp', 'x10')
        """
        if values_str is None:
            values_str = []
        self.name = name
        self.minValue = min_value
        self.maxValue = max_value
        self.type = param_type
        self.denominator = denominator
        self.values_str = values_str

    def correct(self, value):
        """
        Returns the real value of the param
        :param value: value to verify if accomplishes type, min and max due to mutations
        :return: value fixed
        """
        ret = None
        value = int(value)
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

        equals = (self.name == other_param.name and self.minValue == other_param.minValue and
                  self.type == other_param.type and self.denominator == other_param.denominator)
        return equals


class BaseOptimizer(object):
    """
    Abstract class to create optimizer for different machine learning classifier algorithms
    """
    __metaclass__ = ABCMeta

    def __init__(self, features, labels, log_file=None, custom_params={},
                 eval_function=KFoldStratifiedAccuracy):
        """

        :param df: (DataFrame) DataFrame to train and test the classifier
        (maybe in the future this must be change for features, labels list which is more usual)
        """
        self.features = features
        self.labels = labels
        self.custom_params = custom_params
        self.params = self.get_params()
        self.eval_dict = {}
        self.logger = miscellaneous.init_logger(log_file)
        self.eval_function = eval_function

    def init_individual(self, pcls):
        """
        Method to initialize an individual instance

        :param pcls: Method to create the individual as an extension of the class list
        :return: individual
        """
        ps = []
        for k in self.params.keys():
            ps.append(randint(self.params[k].minValue, self.params[k].maxValue))
        ind = pcls(ps)
        return ind

    @abstractmethod
    def individual2dict(self, individual):
        individual_dict = {}
        keys = list(self.params.keys())
        for i in range(len(keys)):
            individual_dict[keys[i]] = individual[i]
        return individual_dict

    @abstractmethod
    def get_params(self):
        """
        Params for the creation of individuals (relative to the algorithm)
        These params define the name of the param, min value, max value, and type

        :return: list of params
        """
        params = {}
        default_params = self.get_default_params()

        for k in default_params.keys():
            if k in self.custom_params:
                params[k] = self.custom_params[k]
            else:
                params[k] = default_params[k]

        # Return all the params
        return params

    @abstractmethod
    def get_clf(self, individual):
        pass

    def get_corrected_clf(self, individual_in):
        individual = copy.copy(individual_in)
        keys = list(self.params.keys())
        for i in range(len(keys)):
            individual[i] = self.params[keys[i]].correct(individual[i])
        return self.get_clf(individual)

    def evaluate_clf(self, individual):
        """
        Method to evaluate the individual, in this case the classifier

        :param individual: individual for evaluation
        :return: mean accuracy, standard deviation accuracy
        """
        # keys = list(self.params.keys())
        # for i in range(len(keys)):
        #    individual[i] = self.params[keys[i]].correct(individual[i])

        # mean, std = KFoldStratifiedAccuracy(self.features, self.labels, self.get_corrected_clf(individual),
        #                                    random_state=1)
        mean, std = self.eval_function(self.features, self.labels, self.get_corrected_clf(individual))
        # out = "Individual evaluation:\n"
        # for i in range(len(self.params)):
        #    out += self.params[i].name + " = " + str(individual[i]) + "\n"
        # out += "  ----> Accuracy: " + str(mean) + " +- " + str(std) + "\n"
        # self.file_out.write(out)
        return mean, std

    def optimize_clf(self, population=10, generations=3):
        """
        Searches through a genetic algorithm the best classifier

        :param int population: Number of members of the first generation
        :param int generations: Number of generations
        :return: Trained classifier
        """
        self.logger.info("Initiating genetic optimization...")
        self.logger.info("Algorithm: {}".format(type(self).__name__))

        # self.file_out.write("Optimizing accuracy:\n")
        # Using deap, custom for decision tree
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Creation of individual and population
        toolbox = base.Toolbox()

        # Paralel
        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)

        toolbox.register("individual", self.init_individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Methods for genetic algorithm
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[x.minValue for x in self.params.values()],
                         up=[x.maxValue for x in self.params.values()], indpb=0.35)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("evaluate", self.evaluate_clf)

        # Tools
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        fpop, logbook = self.customEaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                       ngen=generations, stats=stats,
                                       halloffame=hof)

        self.logger.info("LOGBOOK: \n{}".format(logbook))
        self.logger.info("HALL OF FAME: {} individuals".format(len(hof)))

        for i in range(len(hof)):
            best_score = hof[i].fitness.values[:]
            self.logger.info("Individual TOP {}".format(i + 1))
            self.logger.info("Individual accuracy: {}".format(best_score))
            self.logger.info("Best classifier: {}".format(str(self.get_clf(hof[i]))))


        # self.file_out.write("LOGBOOK: \n"+str(logbook)+"\n")
        # self.file_out.write("Best accuracy: "+str(best_score[0])+"\n")
        # self.file_out.write("Best classifier(without parameter formating(DECIMALS)): "+str(self.get_clf(hof[0])))

        # self.plot_loogbook(logbook=logbook)

        return self.get_corrected_clf(hof[0])

    def plot_loogbook(self, logbook):
        '''
        Plots the given loogboook

        :param logbook: logbook of the genetic algorithm
        '''
        print("TODO")
        # gen = logbook.select("gen")
        # fit_max = logbook.select("max")
        # fit_avg = logbook.select("avg")

    def customEaSimple(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                       halloffame=None, verbose=__debug__):
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
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            self.logger.info(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.logger.info("Generation: {}".format(gen))
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            c = 1
            for ind, fit in zip(invalid_ind, fitnesses):
                self.logger.info("Fitting individual (informational purpose): {}".format(c))
                ind.fitness.values = fit
                c = c + 1

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            for i in range(len(halloffame[:2])):
                best_score = halloffame[i].fitness.values[:]
                self.logger.info("Individual TOP {}".format(i + 1))
                self.logger.info("Individual accuracy: {}".format(best_score))
                self.logger.info("Best classifier: {}".format(str(self.get_clf(halloffame[i]))))
                self.logger.info("Params: {}".format(str(self.get_clf(halloffame[i]).get_params())))

        return population, logbook

#
# fig, ax1 = plt.subplots()
# line1 = ax1.plot(gen, fit_max, "b-", label="Max fit")
# ax1.set_xlabel("Generation")
# ax1.set_ylabel("Fitness", color="b")
#
# line2 = ax1.plot(gen, fit_avg, "r-", label="Avg fit")
#
# lns = line1 + line2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc="lower right")
#
# plt.savefig("optfig")


class TreeOptimizer(BaseOptimizer, ABC):
    """
    Concrete optimizer for sklearn classifier -> sklearn.tree.DecisionTreeClassifier
    """

    def get_clf(self, individual):
        """
        Build a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier sklearn.tree.DecisionTreeClassifier
        """
        individual_dict = self.individual2dict(individual)

        clf = DecisionTreeClassifier(criterion="gini",
                                     class_weight="balanced",
                                     splitter="best",
                                     max_features=None,
                                     max_depth=individual_dict['max_depth'],
                                     min_samples_split=individual_dict['min_samples_split'],
                                     min_samples_leaf=individual_dict['min_samples_leaf'],
                                     min_impurity_decrease=individual_dict['min_impurity_decrease'],
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
    Concrete optimizer for sklearn random forest -> sklearn.ensemble.RandomForestClassifier
    """

    def get_clf(self, individual):
        """
        Builds a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier sklearn.ensemble.RandomForestClassifier
        """
        individual_dict = self.individual2dict(individual)

        clf = RandomForestClassifier(n_estimators=individual_dict['n_estimators'],
                                     criterion="gini",
                                     max_depth=individual_dict['max_depth'],
                                     max_samples=individual_dict['max_samples'],
                                     min_weight_fraction_leaf=0,
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
            "max_features": Param("max_features", 75, 90, float, 100),
            "n_estimators": Param("n_estimators", 50, 250, int),
            "max_depth": Param("max_depth", 3, 40, int),
            "max_samples": Param("max_samples", 20, 50, float, 100)
        }
        return default_params


class ExtraTreesOptimizer(ForestOptimizer, ABC):
    """
    Concrete optimizer for sklearn extra trees -> sklearn.ensemble.ExtraTreesClassifier
    Use the same get_params() as ForestOptimizer
    """

    def get_clf(self, individual):
        """
        Builds a classifier object from an individual one

        :param individual: individual to create a classifier
        :return: classifier ExtraTreesClassifier
        """
        individual_dict = self.individual2dict(individual)

        clf = ExtraTreesClassifier(n_estimators=individual_dict['n_estimators'],
                                   criterion="gini",
                                   max_depth=individual_dict['max_depth'],
                                   min_samples_split=individual_dict['min_samples_split'],
                                   min_samples_leaf=individual_dict['min_samples_leaf'],
                                   min_weight_fraction_leaf=0,
                                   max_features=individual_dict['max_features'],
                                   max_leaf_nodes=None,
                                   bootstrap=False,
                                   oob_score=False,
                                   n_jobs=-1,
                                   random_state=None,
                                   verbose=0,
                                   warm_start=False,
                                   class_weight="balanced")
        return clf


class GradientBoostingOptimizer(ForestOptimizer, ABC):
    """
    Concrete optimizer for sklearn gradient boosting -> sklearn.ensemble.GradientBoostingClassifier
    Use the same get_params() as ForestOptimizer
    """

    def get_params(self):
        """
        Params for the creation of individuals (relative to the algorithm)
        These params define the name of the param, min value, max value, and type

        :return: list of params
        """
        params = super(GradientBoostingOptimizer, self).get_params()
        # learning_rate
        params.append(Param("learning_rate", 1, 10000, float, 1000000))
        # subsample
        params.append(Param("subsample", 0, 100, float, 100))
        # Return all the params
        return params

    def get_clf(self, individual):
        """
        Builds a classifier object from an individual one

        :param individual: individual to create a classifier
        :return: classifier ExtraTreesClassifier
        """
        individual_dict = self.individual2dict(individual)
        clf = GradientBoostingClassifier(n_estimators=individual_dict['n_estimators'],
                                         criterion="friedman_mse",
                                         max_depth=individual_dict['max_depth'],
                                         min_samples_split=individual_dict['min_samples_split'],
                                         min_samples_leaf=individual_dict['min_samples_leaf'],
                                         min_weight_fraction_leaf=0,
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
    Concrete optimizer for extreme gradient boosting classifier -> xgb.XGBRegressor
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
        """
        Build a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier xgb.XGBRegressor
        """
        individual_dict = self.individual2dict(individual)
        clf = xgb.XGBClassifier(base_score=0.5,
                                booster='gbtree',
                                colsample_bytree=individual_dict['colsample_bytree'],
                                colsample_bylevel=1,
                                eval_metric='logloss',
                                use_label_encoder=False,
                                gamma=individual_dict['gamma'],
                                importance_type='gain',
                                learning_rate=individual_dict['learning_rate'],
                                max_delta_step=0,
                                max_depth=individual_dict['max_depth'],
                                min_child_weight=1,
                                missing=None,
                                n_estimators=individual_dict['n_estimators'],
                                n_jobs=-1,
                                nthread=None,
                                objective='binary:logistic',
                                random_state=0,
                                reg_alpha=0,
                                reg_lambda=1,
                                scale_pos_weight=individual_dict['scale_pos_weight'],
                                seed=None,
                                subsample=individual_dict['subsample'],
                                )
        return clf


class CustomXGBClassifierOptimizer(BaseOptimizer, ABC):
    """
    Concrete optimizer for extreme gradient boosting classifier -> using xgb.train
    """

    @staticmethod
    def get_default_params():
        default_params = {
            'eta': Param("eta", 0, 10, float, 10),
            'colsample_bytree': Param("colsample_bytree", 3, 10, float, 10),
            'gamma': Param("gamma", 0, 20, int),
            'max_depth': Param("max_depth", 3, 30, int),
            'subsample': Param("subsample", 700, 1000, float, 1000),
            'scale_pos_weight': Param("scale_pos_weight", 15, 40, float, 100)
        }
        return default_params

    def get_clf(self, individual):
        """
        Build a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier alg_wrapper.CustomXGBClassifier
        """
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
                                  min_child_weight=1,
                                  seed=1,
                                  alpha=0,
                                  scale_pos_weight=individual_dict['scale_pos_weight'])
        return clf


class KerasClassifierOptimizer(BaseOptimizer, ABC):
    """
    Concrete optimizer for KerasClassifier
    """

    @staticmethod
    def get_default_params():
        default_params = {
            'epochs': Param("epochs", 5, 100, int),
            'batch_size': Param("batch_size", 5, 500, int)
        }
        return default_params

    def get_clf(self, individual):
        """
        Build a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier KerasClassifier
        """
        individual_dict = self.individual2dict(individual)
        clf = KerasClassifier(build_fn=generate_model,
                              epochs=individual_dict['epochs'],
                              batch_size=individual_dict['batch_size'])
        return clf


class SVCOptimizer(BaseOptimizer, ABC):
    """
        Concrete optimizer for support vector machine SVC classifier -> sklearn.svm.SVC
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
        """
        Build a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier SVM
        """
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


class MLPOptimizer(BaseOptimizer, ABC):
    """
        Concrete optimizer for support vector machine SVC classifier -> sklearn.svm.SVC
        """

    @staticmethod
    def get_default_params():
        default_params = {
            'learning_rate_init': Param("lr", 1, 6, "nexp"),
            'alpha': Param("alpha", 1, 6, "nexp"),
            'layer1': Param("layer1", 5, 30, "x10"),
            'layer2': Param("layer1", 1, 20, "x10"),
            'layer3': Param("layer1", 1, 10, "x10")
        }
        return default_params

    def get_clf(self, individual):
        """
        Build a classifier object from an individual one

        :param individual: individual to create classifier
        :return: classifier SVM
        """
        individual_dict = self.individual2dict(individual)

        clf = MLPClassifier(activation="relu",
                            solver="adam",
                            learning_rate="constant",
                            hidden_layer_sizes=(individual_dict['layer1'], individual_dict['layer2'],
                                                individual_dict['layer3']),
                            validation_fraction=0.1,
                            early_stopping=True,
                            max_iter=300,
                            learning_rate_init=individual_dict['learning_rate_init'],
                            alpha=individual_dict['alpha'],
                            batch_size=200
                            )
        return clf
