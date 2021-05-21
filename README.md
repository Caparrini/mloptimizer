# mloptimizer

**mloptimizer** is a Python module for hyper-parameters optimization in machine learning using genetic algorithms.


### Installation

```bash
pip install mloptimizer
```
### Quickstart

A simple example of use optimizing hyper-parameters in a decision tree classifier using the iris dataset:

```python
from mloptimizer.genoptimizer import TreeOptimizer
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
opt = TreeOptimizer(X, y, "output_log_file.log")
clf = opt.optimize_clf(10, 10)
```

## Modules used

* [Deap](https://github.com/DEAP/deap) - Genetic Algorithms
* [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting classifier
* [sklearn](https://github.com/scikit-learn/scikit-learn) - Usado para generar RSS

## Wiki

 TODO [Wiki](DOCUMENTATION TODO)

## Authors

* **Antonio Caparrini** - *Owner* - [caparrini](https://github.com/caparrini)

## License

This project is under the [LICENSE](LICENSE) for more details.
