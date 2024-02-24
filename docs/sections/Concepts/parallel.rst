Parallel processing
===================

Relying on the
`Deap capability to parallelize the evaluation of the fitness function
<https://deap.readthedocs.io/en/master/tutorials/basic/part4.html>`__,
we can use the ``multiprocessing`` module to parallelize the evaluation of the fitness function.
This is done passing the ``use_parallel`` parameter as ``True`` to initialize the ``Optimizer`` object.
This parameter is set to ``False`` by default.

An example of the speedup that can be achieved using parallel processing is shown below.

.. note::
   In the example below, the seed is set to 25 to ensure the result using parallel processing is the same as the one without parallel processing.

.. warning::
   Parallel processing is not supported for the ``XGB`` and ``Keras`` classifiers.

.. code-block:: python

    from sklearn.datasets import load_iris
    import time

    from mloptimizer.genoptimizer import TreeOptimizer

    X, y = dataset(return_X_y=True)
    my_seed = 25
    population = 50
    generations = 4

    opt_with_parallel = optimizer(X, y, seed=my_seed, use_parallel=True)

    start_time_parallel = time.time()
    clf_with_parallel = opt_with_parallel.optimize_clf(population, generations)
    end_time_parallel = time.time()

    opt = optimizer(X, y, seed=my_seed, use_parallel=False)

    start_time = time.time()
    clf = opt.optimize_clf(population, generations)
    end_time = time.time()

    elapsed_time_parallel = end_time_parallel - start_time_parallel
    elapsed_time = end_time - start_time
    speedup = round(((elapsed_time_parallel / elapsed_time) - 1) * 100, 2)

    print(f"Elapsed time with parallel: {elapsed_time_parallel}")
    print(f"Elapsed time without parallel: {elapsed_time}")
    print(f"Speedup: {speedup}%")

