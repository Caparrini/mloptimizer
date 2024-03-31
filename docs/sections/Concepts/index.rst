Concepts
==================

Concepts are the building blocks of the hyperparameter optimization
framework. They are used to define the search space and the score function.

.. mermaid::

   classDiagram
       class Optimizer{
         +estimator_class estimator_class
         +HyperparameterSpace hyperspace
         +Tracker tracker
         +Evaluator evaluator
         +IndividualUtils individual_utils
         optimize_clf()
       }
       class HyperparameterSpace{
         +dict fixed_hyperparams
         +dict evolvable_hyperparams
         from_json()
         to_json()
       }
       class Evaluator{
         evaluate()
         evaluate_individual()
       }
       class IndividualUtils{
         individual2dict()
         get_clf()
       }
       Optimizer "1" --o "1" HyperparameterSpace
       Optimizer "1" --o "1" Evaluator
       Optimizer "1" --o "1" IndividualUtils


.. toctree::
   :hidden:

   hyperparam
   score_functions
   reproducibility
   parallel
