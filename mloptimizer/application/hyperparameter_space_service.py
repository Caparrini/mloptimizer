from mloptimizer.infrastructure.repositories import HyperparameterSpaceRepository
from mloptimizer.domain.hyperspace import HyperparameterSpace
import os


class HyperparameterSpaceService:
    """
    Service to manage hyperparameter spaces.
    """

    def __init__(self, base_config_path=None):
        if base_config_path is None:
            base_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "infrastructure", "config", "hyperspace")
        self.base_config_path = base_config_path

    def load_hyperparameter_space(self, estimator_class):
        """
        Load a hyperparameter space for an estimator from a JSON file.
        """
        file_name = f"{estimator_class.__name__.lower()}_hyperparameter_space.json"
        file_path = os.path.join(self.base_config_path, file_name)

        # Delegate to repository
        hyperparam_data = HyperparameterSpaceRepository.load_json(file_path)

        return HyperparameterSpace.from_json_data(hyperparam_data)

    def save_hyperparameter_space(self, hyperparam_space: HyperparameterSpace, estimator_class, overwrite=False):
        """
        Save a hyperparameter space for an estimator to a JSON file.
        """
        file_name = f"{estimator_class.__name__.lower()}_hyperparameter_space.json"
        file_path = os.path.join(self.base_config_path, file_name)

        # Delegate to repository
        hyperparam_data = {
            'fixed_hyperparams': hyperparam_space.fixed_hyperparams,
            'evolvable_hyperparams': {k: v.__dict__ for k, v in hyperparam_space.evolvable_hyperparams.items()}
        }
        HyperparameterSpaceRepository.save_json(hyperparam_data, file_path, overwrite)
