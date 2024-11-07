Features
========

`mloptimizer` is designed to streamline hyperparameter optimization for machine learning models by leveraging genetic algorithms. With a flexible and extensible architecture, it integrates seamlessly with the :mod:`scikit-learn` API, making it a valuable tool for both researchers and practitioners looking to improve model performance efficiently. Below is an overview of key and advanced features that make `mloptimizer` a robust choice for hyperparameter tuning.

Key Features
------------

- **User-Friendly**: Intuitive syntax, fully compatible with the :mod:`scikit-learn` API.
- **DEAP-Based Genetic Algorithms**: Built on the :mod:`deap` library, which supports flexible and robust genetic search algorithms. The use of :mod:`deap` provides a foundation for effective evolutionary computation techniques within `mloptimizer`.
- **Predefined and Custom Hyperparameter Spaces**: Includes default hyperparameter spaces for commonly used algorithms, along with options to define custom spaces to suit unique needs.
- **Customizable Score Functions**: Offers default metrics for model evaluation, with the flexibility to add custom scoring functions.
- **Reproducibility and Parallelization**: Ensures reproducible results and supports parallel processing to accelerate optimization tasks.

Advanced Features
-----------------

- **Extensibility**: Easily extendable to additional machine learning models that comply with the :class:`Estimator <sklearn.base.Estimator>` class from the :mod:`scikit-learn` API.
- **Custom Hyperparameter Ranges**: Allows users to define specific hyperparameter ranges as needed.
- **MLflow Integration** (Optional): Enables tracking of optimization runs through :mod:`mlflow` for more detailed analysis.
- **Optimization Monitoring**: Provides detailed logs and visualizations to monitor the optimization process.
- **Checkpointing and Resuming**: Supports checkpointing to save the state of the optimization process and resume from a specific point if needed.
- **Search Space Visualization**: Generates visual representations of the search space to aid in understanding the hyperparameter landscape.