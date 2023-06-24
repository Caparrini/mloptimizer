import pytest

from mloptimizer.genoptimizer import Param


@pytest.fixture
def int_param():
    return Param('int_param', 1, 10, int)


@pytest.fixture
def float_param():
    return Param('float_param', 1, 200, float, denominator=100)


@pytest.fixture
def nexp_param():
    return Param('nexp_param', 0, 10, 'nexp')


@pytest.fixture
def x10_param():
    return Param('x10_param', 0, 10, 'x10')


def test_int_param_correct(int_param):
    assert int_param.correct(5) == 5
    assert int_param.correct(0) == 1
    assert int_param.correct(11) == 10


def test_float_param_correct(float_param):
    assert float_param.correct(50) == 0.5
    assert float_param.correct(300) == 2.0
    assert float_param.correct(150) == 1.5


def test_nexp_param_correct(nexp_param):
    assert nexp_param.correct(0) == 1
    assert nexp_param.correct(10) == 0.0000000001
    assert nexp_param.correct(11) == 0.0000000001


def test_x10_param_correct(x10_param):
    assert x10_param.correct(5) == 50
    assert x10_param.correct(-1) == 0
    assert x10_param.correct(11) == 100


def test_str_float_param(float_param):
    float_param_str = str(float_param)
    assert float_param_str == "Param('float_param', 1, 200, float, 100)"
