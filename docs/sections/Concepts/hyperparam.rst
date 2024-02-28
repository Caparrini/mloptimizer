====================
Hyperparam Class
====================

The Hyperparam class is a crucial component of our library, designed to optimize the hyperparameters of machine learning algorithms using the DEAP library, which provides genetic algorithms.

Why We Need the Hyperparam Class
--------------------------------

In the context of genetic optimization, a common problem is the repeated evaluation of slightly different individuals. This can lead to inefficiencies in the optimization process. To mitigate this, we use the Hyperparam class to limit the values of the hyperparameters, ensuring that the same individuals are not evaluated multiple times.

How It Is Used
--------------

The Hyperparam class is used to define a hyperparameter to optimize. It includes the name, minimum value, maximum value, and type of the hyperparameter. This class also controls the precision of the hyperparameter to avoid multiple evaluations with close values due to decimal positions.

The Hyperparam class has several methods, including:

- `__init__`: Initializes a new instance of the Hyperparam class.
- `correct`: Returns the real value of the hyperparameter in case some mutation could surpass the limits.
- `__eq__`: Overrides the default implementation to compare two Hyperparam instances.
- `__str__` and `__repr__`: Overrides the default implementations to provide a string representation of the Hyperparam instance.

Types of Hyperparam
-------------------

The `Hyperparam` class supports several types of hyperparameters. Here are examples of each type:

- Integer hyperparameter:

.. code-block:: python

   hyperparam_int = Hyperparam(name='max_depth', min_value=1,
                               max_value=10, hyperparam_type='int')


- Float hyperparameter:

.. code-block:: python

    hyperparam_float = Hyperparam(name='learning_rate', min_value=0.01, max_value=1.0,
                                  hyperparam_type='float', scale=100)


- 'nexp' hyperparameter:

.. code-block:: python

    hyperparam_nexp = Hyperparam(name='nexp_param', min_value=1,
                                 max_value=100, hyperparam_type='nexp')

- 'x10' hyperparameter:

.. code-block:: python

    hyperparam_x10 = Hyperparam(name='x10_param', min_value=1,
                                max_value=100, hyperparam_type='x10')

In these examples, we define hyperparameters of different types. The 'nexp' and 'x10' types are special types that apply a transformation to the hyperparameter value.

Examples
--------

Here's an example of how to use the Hyperparam class:

.. code-block:: python

   # Define a hyperparameter
   hyperparam = Hyperparam(name='learning_rate', min_value=0, max_value=1,
                           hyperparam_type='float', scale=100)

   # Correct a value
   # This will return 1.0 as 150 is beyond the max_value
   corrected_value = hyperparam.correct(150)


In this example, we define a hyperparameter named 'learning_rate' with a minimum value of 0, a maximum value of 1, and a type of float. The 'correct' method is then used to correct a value that is beyond the defined maximum value.

Here's an example of how you can create a `HyperparameterSpace` instance and pass custom hyperparameters to it:

.. code-block:: python

   from mloptimizer.hyperparams import Hyperparam, HyperparameterSpace

   # Define custom hyperparameters
   fixed_hyperparams = {
       "criterion": "gini"
   }
   evolvable_hyperparams = {
       "min_samples_split": Hyperparam("min_samples_split", 2, 50, 'int'),
       "min_samples_leaf": Hyperparam("min_samples_leaf", 1, 20, 'int'),
       "max_depth": Hyperparam("max_depth", 2, 20, 'int'),
       "min_impurity_decrease": Hyperparam("min_impurity_decrease", 0, 150, 'float', 1000),
       "ccp_alpha": Hyperparam("ccp_alpha", 0, 300, 'float', 100000)
   }


   # Create a HyperparameterSpace instance
   hyperparam_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)

   # Then we can use the hyperparam_space instance to optimize the hyperparameters
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.datasets import load_iris
   from mloptimizer.genoptimizer import SklearnOptimizer

   # Load the iris dataset
   X,y = load_iris(return_X_y=True)

   tree_optimizer = SklearnOptimizer(clf_class=DecisionTreeClassifier,
                                    hyperparam_space=hyperparam_space,
                                    features=X, labels=y)
   tree_optimizer.optimize_clf(3, 3)


In this example, we define custom hyperparameters and create a `HyperparameterSpace` instance. We then use the `HyperparameterSpace` instance to optimize the hyperparameters of a `DecisionTreeClassifier` using the `SklearnOptimizer` class.
