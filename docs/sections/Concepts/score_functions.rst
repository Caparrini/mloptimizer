=============================
Score Functions (NEED UPDATE)
=============================

The `model_evaluation.py` module in our library provides several score functions that are used to evaluate the performance of machine learning algorithms. These score functions are crucial in the context of genetic optimization, where they serve as fitness values. In genetic optimization, a fitness value determines how well an individual (in this case, a machine learning algorithm defined by its hyperparameters) performs in a given generation. The better the fitness value, the more likely the individual is to survive and reproduce in the next generation.

A score function takes as input:
- The true labels of the data
- The predicted labels of the data
- A machine learning algorithm complying with the scikit-learn API
- A scoring function metric (e.g. accuracy, precision, recall, F1 score, etc.)

.. note::
   The library provides several score functions that can be used for genetic optimization. However, users can also define their own score functions if they wish to do so.

Score Functions
---------------

The `model_evaluation.py` module provides the following score functions:

- `train_score`: This function trains a classifier with the provided features and labels, and then calculates a score using the provided score function.

- `train_test_score`: This function splits the provided features and labels into a training set and a test set. It then trains a classifier on the training set and calculates a score on the test set using the provided score function.

- `kfold_score`: This function evaluates a classifier using K-Fold cross-validation. It splits the provided features and labels into K folds, trains a classifier on K-1 folds, and calculates a score on the remaining fold. This process is repeated K times, and the function returns the average score across all folds.

- `kfold_stratified_score`: This function is similar to `kfold_score`, but it uses stratified K-Fold cross-validation. This means that it preserves the percentage of samples for each class in each fold.

- `temporal_kfold_score`: This function is similar to `kfold_score`, but it uses temporal K-Fold cross-validation. This means that it respects the order of the data, making it suitable for time series data.

Each of these score functions takes a classifier, features, and labels as input. They also take a score function as input, which is used to calculate the score. The score function could be any function that takes the true labels and the predicted labels as input and returns a score. Examples of score functions include accuracy, precision, recall, F1 score, etc.

Examples
--------

Here's an example of how to use the `train_score` function:

.. code-block:: python

   from mloptimizer.evaluation import model_evaluation
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score

   # Define features, labels, and classifier
   features = [...]
   labels = [...]
   clf = RandomForestClassifier()

   # Use the train_score function
   score = model_evaluation.train_score(features, labels, clf, score_function=accuracy_score)


In this example, we first define the features, labels, and classifier. We then use the `train_score` function to train the classifier and calculate the score. The `accuracy_score` function from `sklearn.metrics` is used as the score function.

