from mloptimizer.domain.hyperspace import HyperparameterSpace
from mloptimizer.domain.hyperspace import Hyperparam
from mloptimizer.application import HyperparameterSpaceService

class HyperparameterSpaceBuilder:
    def __init__(self):
        self.fixed_hyperparams = {}
        self.evolvable_hyperparams = {}
        self.service = HyperparameterSpaceService()

    def add_int_param(self, name, min_value, max_value):
        param = Hyperparam(name=name, min_value=min_value, max_value=max_value, hyperparam_type='int')
        self.evolvable_hyperparams[name] = param
        return self

    def add_float_param(self, name, min_value, max_value, scale=100):
        param = Hyperparam(name=name, min_value=min_value, max_value=max_value, hyperparam_type='float', scale=scale)
        self.evolvable_hyperparams[name] = param
        return self

    def add_categorical_param(self, name, values):
        param = Hyperparam.from_values_list(name=name, values_str=values)
        self.evolvable_hyperparams[name] = param
        return self

    def set_fixed_param(self, name, value):
        self.fixed_hyperparams[name] = value
        return self

    def build(self):
        return HyperparameterSpace(fixed_hyperparams=self.fixed_hyperparams, evolvable_hyperparams=self.evolvable_hyperparams)

    def load_default_space(self, estimator_class):
        """Load a default hyperparameter space using the application service."""
        return self.service.load_hyperparameter_space(estimator_class)

    def save_space(self, hyperparam_space, estimator_class, overwrite=False):
        """Save the hyperparameter space using the application service."""
        self.service.save_hyperparameter_space(hyperparam_space, estimator_class, overwrite)