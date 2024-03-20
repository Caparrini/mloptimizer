Concepts
==================

Concepts are the building blocks of the hyperparameter optimization
framework. They are used to define the search space and the score function.

.. mermaid::

   classDiagram
       class Optimizer{
         +estimator_class estimator_class
       }


.. toctree::
   :hidden:

   hyperparam
   score_functions
   reproducibility
   parallel
