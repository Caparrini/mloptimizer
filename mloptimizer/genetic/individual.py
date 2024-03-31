from mloptimizer.hyperparams import HyperparameterSpace


class IndividualUtils:
    def __init__(self, hyperparam_space: HyperparameterSpace = None, estimator_class=None, mlopt_seed=None):
        self.hyperparam_space = hyperparam_space
        self.estimator_class = estimator_class
        self.mlopt_seed = mlopt_seed

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        clf = self.estimator_class(random_state=self.mlopt_seed, **individual_dict)
        return clf

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
