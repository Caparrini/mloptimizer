import pytest
import json
import os
from mloptimizer.domain.hyperspace import HyperparameterSpace, Hyperparam


@pytest.fixture
def hyperparam_space():
    fixed_hyperparams = {'fixed_param_1': 1, 'fixed_param_2': 2}
    evolvable_hyperparams = {
        'evolvable_param_1': Hyperparam(name='evolvable_param_1', min_value=1, max_value=2, hyperparam_type='int'),
        'evolvable_param_2': Hyperparam(name='evolvable_param_2', min_value=2, max_value=8, hyperparam_type='int')
    }
    return HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)


@pytest.fixture
def json_file_path(tmpdir):
    return tmpdir.join("test_hyperparameter_space.json")


def test_from_json(hyperparam_space, json_file_path):
    # Save data to file
    hyperparam_space.to_json(str(json_file_path), overwrite=True)

    # Load from file
    loaded_hyperparam_space = HyperparameterSpace.from_json(str(json_file_path))

    # Assertions
    assert isinstance(loaded_hyperparam_space, HyperparameterSpace)
    assert loaded_hyperparam_space.fixed_hyperparams == hyperparam_space.fixed_hyperparams
    assert loaded_hyperparam_space.evolvable_hyperparams.keys() == hyperparam_space.evolvable_hyperparams.keys()


def test_to_json(hyperparam_space, json_file_path):
    # Test if the method correctly writes the instance to a JSON file
    hyperparam_space.to_json(str(json_file_path), overwrite=True)

    # Check if the file was correctly written
    with open(str(json_file_path), 'r') as file:
        data = json.load(file)

    assert 'fixed_hyperparams' in data
    assert data['fixed_hyperparams'] == hyperparam_space.fixed_hyperparams
    assert 'evolvable_hyperparams' in data
    assert (data['evolvable_hyperparams'] ==
            {key: value.__dict__ for key, value in hyperparam_space.evolvable_hyperparams.items()})

    # Test that FileExistsError is raised if overwrite=False
    with pytest.raises(FileExistsError):
        hyperparam_space.to_json(str(json_file_path), overwrite=False)
