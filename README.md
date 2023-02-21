# mloptimizer

**mloptimizer** is a Python module for hyperparameter optimization in machine learning using genetic algorithms.

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


### Quickstart

Here's a simple example of how to optimize hyperparameters in a decision tree classifier using the iris dataset:

```python
from mloptimizer.genoptimizer import TreeOptimizer
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# The log file will be created in the current directory and will have info about optimizations performed
opt = TreeOptimizer(X, y, "output_log_file.log")

clf = opt.optimize_clf(10, 10)
```

The las line of code will create a directory in the current folder with a name similar to `YYYYMMDD_nnnnnnnnnn_TreeOptimizer`.
This folder contains a backup file for each generation and an `opt.log` inform with all the steps, the best estimator and the result of the optimization.

## Dependencies

The following dependencies are used in `mloptimizer`:

* [Deap](https://github.com/DEAP/deap) - Genetic Algorithms
* [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting classifier
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) - Usado para generar RSS

## Documentation

The documentation for `mloptimizer` can be found in the project's [wiki](DOCUMENTATION TODO)

## Authors

* **Antonio Caparrini** - *Owner* - [caparrini](https://github.com/caparrini)

## License

This project is licensed under the [MIT License](LICENSE).
