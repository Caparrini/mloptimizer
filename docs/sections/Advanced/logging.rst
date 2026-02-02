Logging and Verbosity
=====================

``mloptimizer`` uses Python's standard logging module and provides a ``verbose`` parameter
for easy control of output verbosity.

Verbose Parameter
-----------------

The simplest way to control logging output is through the ``verbose`` parameter in
:class:`GeneticSearch <mloptimizer.interfaces.GeneticSearch>`:

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
    from sklearn.tree import DecisionTreeClassifier

    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Silent mode (default) - no logging output
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        verbose=0  # Default
    )

    # Info level - optimization lifecycle and generation summaries
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        verbose=1
    )

    # Debug level - detailed evaluation information
    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        verbose=2
    )

**Verbose levels:**

- ``verbose=0``: Silent mode. No logging output (default).
- ``verbose=1``: INFO level. Shows optimization start/end, generation summaries, and early stopping messages.
- ``verbose=2``: DEBUG level. Shows detailed evaluation info, MLflow operations, and internal state.

This is similar to how scikit-learn and XGBoost handle verbosity.

Advanced Logging Configuration
------------------------------

For advanced use cases, you can configure Python's logging module directly. This is useful when:

- Integrating ``mloptimizer`` into a larger application with existing logging
- Redirecting logs to files
- Filtering specific modules

Basic Configuration
~~~~~~~~~~~~~~~~~~~

To enable logging output, configure Python's logging module before running your optimization:

.. code-block:: python

    import logging

    # Enable INFO level logging globally
    logging.basicConfig(level=logging.INFO)

    # Now use mloptimizer
    from mloptimizer.interfaces import GeneticSearch
    # ... your optimization code

Configuring mloptimizer's Logger Only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable logging specifically for mloptimizer without affecting other libraries:

.. code-block:: python

    import logging

    # Get mloptimizer's logger
    mlopt_logger = logging.getLogger("mloptimizer")
    mlopt_logger.setLevel(logging.INFO)

    # Add a handler to output to console
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    mlopt_logger.addHandler(handler)

Logging to a File
~~~~~~~~~~~~~~~~~

To save logs to a file for later analysis:

.. code-block:: python

    import logging

    # Configure logging to file
    logging.basicConfig(
        filename='optimization.log',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s"
    )

Or configure both console and file output:

.. code-block:: python

    import logging

    # Create logger
    logger = logging.getLogger("mloptimizer")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler("optimization.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    logger.addHandler(file_handler)

Key Loggers
-----------

The main loggers used by ``mloptimizer``:

- ``mloptimizer.infrastructure.tracking.tracker`` - Optimization lifecycle, MLflow integration
- ``mloptimizer.domain.optimization.genetic_algorithm`` - GA operations, early stopping
- ``mloptimizer.infrastructure.util.utils`` - File/folder operations

Log Levels
----------

mloptimizer uses standard Python logging levels:

- **DEBUG**: Detailed information for diagnosing problems (individual evaluations, internal state)
- **INFO**: Confirmation that things are working as expected (optimization start/end, generation summaries)
- **WARNING**: Indication of potential issues (deprecated parameters, suboptimal configurations)
- **ERROR**: Serious problems that prevent operation

Silencing Logs
--------------

To completely suppress mloptimizer's log output:

.. code-block:: python

    import logging
    logging.getLogger("mloptimizer").setLevel(logging.CRITICAL)

Or to only see warnings and errors:

.. code-block:: python

    import logging
    logging.getLogger("mloptimizer").setLevel(logging.WARNING)

Example: Complete Logging Setup
-------------------------------

.. code-block:: python

    import logging
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

    # Setup file logging
    logging.basicConfig(
        filename='mloptimizer.log',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    X, y = load_iris(return_X_y=True)
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        generations=5,
        population_size=10,
        verbose=1  # Enable INFO level logging
    )
    opt.fit(X, y)

    # Logs will be written to mloptimizer.log

.. note::

    The ``verbose`` parameter only affects mloptimizer's own loggers. Third-party libraries
    (sklearn, DEAP, etc.) have their own logging configuration.

Integration with scikit-learn
-----------------------------

Since mloptimizer follows the same logging pattern as scikit-learn and other major Python libraries, you can configure logging for your entire ML pipeline consistently:

.. code-block:: python

    import logging

    # Configure logging for all ML libraries
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Now all libraries (mloptimizer, sklearn, etc.) will log consistently
