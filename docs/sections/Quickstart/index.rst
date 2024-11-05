Quick Start
==================

This Quick Start guide will introduce you to the core steps for hyperparameter optimization using `mloptimizer`. By leveraging `GeneticSearch` and `HyperparameterSpaceBuilder`, you’ll gain control over tuning your model’s performance with a streamlined approach similar to `GridSearchCV` from :mod:`scikit-learn`.

.. toctree::
   step1
   step2
   step3
   step4

Overview of Steps
-----------------

1. **Step 1: Setting Up an Optimization with GeneticSearch**
   Begin by setting up `GeneticSearch` as your optimization engine. This step will show you how to initialize `GeneticSearch`, configure the genetic algorithm parameters, and use it seamlessly within machine learning pipelines, following a familiar approach to `GridSearchCV`.

2. **Step 2: Defining Hyperparameter Spaces with HyperparameterSpaceBuilder**
   Define your search space using `HyperparameterSpaceBuilder`. Learn how to create flexible, robust hyperparameter spaces with fixed and evolvable parameters, either through default setups or custom configurations tailored to your model’s needs.

3. **Step 3: Running and Monitoring Optimization**
   Execute and monitor the optimization process with `GeneticSearch`. This step guides you through running the optimization and tracking progress by observing key metrics, helping you understand the performance of each generation.

4. **Step 4: Reviewing and Interpreting Results**
   Finally, assess the outcomes of the optimization process. This step explains how to identify the best estimator, analyze key performance indicators, and interpret results with practical examples to make data-driven adjustments.
