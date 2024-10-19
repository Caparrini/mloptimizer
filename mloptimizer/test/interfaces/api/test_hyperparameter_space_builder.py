# mloptimizer/test/interfaces/api/test_hyperparameter_space_builder.py

import pytest
from mloptimizer.domain.hyperspace import HyperparameterSpace, Hyperparam
from mloptimizer.application import HyperparameterSpaceService
from mloptimizer.interfaces.api import HyperparameterSpaceBuilder
from mloptimizer.infrastructure.repositories import HyperparameterSpaceRepository


@pytest.fixture
def mock_service(mocker):
    return mocker.patch('mloptimizer.application.HyperparameterSpaceService')


def test_add_int_param(mock_service):
    builder = HyperparameterSpaceBuilder()
    builder.add_int_param("param1", 1, 10)
    assert "param1" in builder.evolvable_hyperparams


def test_add_float_param(mock_service):
    builder = HyperparameterSpaceBuilder()
    builder.add_float_param("param2", 0.1, 1.0)
    assert "param2" in builder.evolvable_hyperparams


def test_add_categorical_param(mock_service):
    builder = HyperparameterSpaceBuilder()
    builder.add_categorical_param("param3", ["a", "b", "c"])
    assert "param3" in builder.evolvable_hyperparams


def test_set_fixed_param(mock_service):
    builder = HyperparameterSpaceBuilder()
    builder.set_fixed_param("fixed_param1", 5)
    assert "fixed_param1" in builder.fixed_hyperparams


def test_build(mock_service):
    builder = HyperparameterSpaceBuilder()
    builder.add_int_param("param1", 1, 10)
    builder.set_fixed_param("fixed_param1", 5)
    space = builder.build()

    assert isinstance(space, HyperparameterSpace)
    assert space.fixed_hyperparams["fixed_param1"] == 5
    assert space.evolvable_hyperparams["param1"].name == "param1"


def test_load_default_space(mock_service, mocker):
    mock_service.load_hyperparameter_space.return_value = HyperparameterSpace({}, {})
    mocker.patch('mloptimizer.infrastructure.repositories.HyperparameterSpaceRepository.load_json', return_value={
        'fixed_hyperparams': {},
        'evolvable_hyperparams': {}
    })

    builder = HyperparameterSpaceBuilder()
    result = builder.load_default_space(estimator_class=object)

    assert isinstance(result, HyperparameterSpace)


def test_save_space(mock_service, mocker):
    space = HyperparameterSpace({}, {})
    mocker.patch('mloptimizer.infrastructure.repositories.HyperparameterSpaceRepository.save_json')

    builder = HyperparameterSpaceBuilder()
    builder.save_space(space, estimator_class=object, overwrite=True)

    HyperparameterSpaceRepository.save_json.assert_called_once()