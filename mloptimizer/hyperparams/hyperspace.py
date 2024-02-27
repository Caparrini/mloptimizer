import json
from .hyperparam import Hyperparam
import os


class HyperparameterSpace:
    default_hyperparameter_spaces_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "conf", "default_hyperparameter_spaces.json")

    def __init__(self, fixed_hyperparams, evolvable_hyperparams):
        self.fixed_hyperparams = fixed_hyperparams
        self.evolvable_hyperparams = evolvable_hyperparams

    @classmethod
    def from_json(cls, file_path):
        try:
            # Load JSON data from the file
            with open(file_path, 'r') as file:
                json_data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError("The file {file_path} is not a valid JSON file", json_data, e.pos)

        # Extract fixed hyperparams, just a dictionary
        fixed_hyperparams = json_data['fixed_hyperparams']
        # Extract evolvable hyperparams
        evolvable_hyperparams_dict = json_data['evolvable_hyperparams']
        evolvable_hyperparams = {}
        for k in evolvable_hyperparams_dict.keys():
            evolvable_hyperparams[k] = Hyperparam(**evolvable_hyperparams_dict[k])

        return cls(fixed_hyperparams=fixed_hyperparams, evolvable_hyperparams=evolvable_hyperparams)

    def to_json(self, file_path, overwrite=False):
        # Verify the file path exists
        if file_path is None:
            raise ValueError("The file path must be a valid path")
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"The file path {file_path} exist and overwrite is {overwrite}")
        # Create a dictionary with the fixed hyperparams and evolvable hyperparams
        hyperparams_dict = {
            'fixed_hyperparams': self.fixed_hyperparams,
            'evolvable_hyperparams': {key: value.__dict__ for key, value in self.evolvable_hyperparams.items()}
        }

        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(hyperparams_dict, file, indent=4)

    @staticmethod
    def get_default_hyperparameter_space(clf_class):
        """
        This method returns a dictionary with the default hyperparameters for the scikit-learn classifier.
        It reads the default_hyperparameter_spaces.json file and returns the hyperparameters for the classifier
        """
        with open(HyperparameterSpace.default_hyperparameter_spaces_json, 'r') as file:
            default_hyperparams = json.load(file)
        if clf_class.__name__ in default_hyperparams.keys():
            return HyperparameterSpace.from_json(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "conf", default_hyperparams[clf_class.__name__]
                             )
            )
        else:
            raise ValueError(f"Default hyperparameter space for {clf_class.__name__} not found")

    def __str__(self):
        return (f"HyperparameterSpace(fixed_hyperparams={self.fixed_hyperparams}, "
                f"evolvable_hyperparams={self.evolvable_hyperparams})")

    def __repr__(self):
        return (f"HyperparameterSpace(fixed_hyperparams={self.fixed_hyperparams}, "
                f"evolvable_hyperparams={self.evolvable_hyperparams})")
