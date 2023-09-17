[![Documentation Status](https://readthedocs.org/projects/mloptimizer/badge/?version=latest)](https://mloptimizer.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/mloptimizer.svg)](https://badge.fury.io/py/mloptimizer)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mloptimizer.svg)](https://pypi.python.org/pypi/mloptimizer/)
[![Coverage Status](http://codecov.io/github/Caparrini/mloptimizer/coverage.svg?branch=master)](https://codecov.io/gh/Caparrini/mloptmizer/branch/master)

# mloptimizer

**mloptimizer** is a Python library for optimizing hyperparameters of machine learning algorithms using genetic algorithms. With mloptimizer, you can find the optimal set of hyperparameters for a given machine learning model and dataset, which can significantly improve the performance of the model. The library supports several popular machine learning algorithms, including decision trees, random forests, and gradient boosting classifiers. The genetic algorithm used in mloptimizer provides an efficient and flexible approach to search for the optimal hyperparameters in a large search space.

### Installation

It is recommended to create a virtual environment using the `venv` package. 
To learn more about how to use `venv`, 
check out the official Python documentation at 
https://docs.python.org/3/library/venv.html.

```bash
# Create the virtual environment
python -m venv myenv
# Activate the virtual environment
source myenv/bin/activate
```

To install `mloptimizer`, run:

```bash
pip install mloptimizer
```

You can get more information about the package installation at https://pypi.org/project/mloptimizer/.


### Quickstart

Here's a simple example of how to optimize hyperparameters in a decision tree classifier using the iris dataset:

```python
from mloptimizer.genoptimizer import TreeOptimizer
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

#   
opt = TreeOptimizer(X, y, "Optimizer")

clf = opt.optimize_clf(10, 10)
```

The last line of code will create a directory in the current folder with a name like `YYYYMMDD_nnnnnnnnnn_TreeOptimizer`.
This folder contains the results of the optimization, 
including the best estimator found and the log file `opt.log` informing with all the steps, 
the best estimator and the result of the optimization.

More details in the [documentation](http://mloptimizer.readthedocs.io/).

## Dependencies

The following dependencies are used in `mloptimizer`:

* [Deap](https://github.com/DEAP/deap) - Genetic Algorithms
* [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting classifier
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) - Machine learning algorithms and utilities
* [Keras](https://keras.io) - Deep learning library

## Documentation

The documentation for `mloptimizer` can be found in the project's [wiki](http://mloptimizer.readthedocs.io/)
with examples and classes and methods reference.

## Authors

* **Antonio Caparrini** - *Owner* - [caparrini](https://github.com/caparrini)

## License

This project is licensed under the [MIT License](LICENSE).
