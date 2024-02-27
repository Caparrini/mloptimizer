import pytest
import json
import os
from mloptimizer.hyperparams import HyperparameterSpace, Hyperparam


@pytest.fixture
def hyperparam_space():
    fixed_hyperparams = {'fixed_param_1': 1, 'fixed_param_2': 2}
    evolvable_hyperparams = {'evolvable_param_1': Hyperparam(name='evolvable_param_1', min_value=1,
                                                             max_value=2, hyperparam_type='int'),
                             'evolvable_param_2': Hyperparam(name='evolvable_param_2', min_value=2,
                                                             max_value=8, hyperparam_type='int')}
    return HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)


@pytest.fixture
def json_file_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "conf", "DecisionTreeClassifier_default_HyperparamSpace.json")


def test_from_json(json_file_path):
    # Test if the method correctly creates an instance from a JSON file
    i_hyperparam_space = HyperparameterSpace.from_json(json_file_path)
    assert isinstance(i_hyperparam_space, HyperparameterSpace)


def test_to_json(hyperparam_space):
    # Test if the method correctly writes the instance to a JSON file
    aux_json_file_path = "aux.json"
    hyperparam_space.to_json(aux_json_file_path, overwrite=True)
    with open(aux_json_file_path, 'r') as file:
        data = json.load(file)
    try:
        hyperparam_space.to_json(aux_json_file_path, overwrite=False)
        raise AssertionError(f"The method should raise a FileExistsError")
    except FileExistsError:
        pass
    assert 'fixed_hyperparams' in data
    assert data['fixed_hyperparams'] == hyperparam_space.fixed_hyperparams
    assert 'evolvable_hyperparams' in data
    assert (data['evolvable_hyperparams'] ==
            {key: value.__dict__ for key, value in hyperparam_space.evolvable_hyperparams.items()})
    os.remove(aux_json_file_path)
