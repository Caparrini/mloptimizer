import logging
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, \
    train_test_split, KFold


def train_score(features, labels, clf, score_function=accuracy_score):
    """
    Trains the classifier with the features and labels.

    Parameters
    ----------
    features : list
        List of features
    labels : list
        List of labels
    clf : object
        classifier with methods fit, predict and score
    score_function : func
        function that receives y, y_pred and return a score

    Returns
    -------
    accuracy : float
        score of the classifier
    """
    logging.info("Score metric over training data\nClassifier:{}\nscore_metric:{}".format(clf, score_function))
    clf.fit(features, labels)
    predictions = clf.predict(features)
    accuracy = score_function(labels, predictions)
    logging.info("Accuracy: {:.3f}".format(round(accuracy, 3)))
    return accuracy


def train_test_score(features, labels, clf, score_function=accuracy_score, test_size=0.2, random_state=None):
    """
    Trains the classifier with the train set features and labels,
    then uses the test features and labels to create score.

    Parameters
    ----------
    features : list
        List of features
    labels : list
        List of labels
    clf : object
        Classifier with methods fit, predict, and score
    score_function : func, optional
        Function that receives y_true and y_pred and returns a score
    test_size : float, optional
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split

    Returns
    -------
    accuracy : float
        Score of the classifier on the test set
    """
    # Splitting the dataset into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    # Training the classifier
    clf.fit(features_train, labels_train)

    # Making predictions on the test set
    predictions = clf.predict(features_test)

    # Calculating the accuracy
    accuracy = score_function(labels_test, predictions)

    logging.info("Score metric over test data\nClassifier:{}\nscore_metric:{}".format(clf, score_function))
    logging.info("Accuracy: {:.3f}".format(round(accuracy, 3)))

    return accuracy


def kfold_score(features, labels, clf, score_function=accuracy_score, n_splits=5, random_state=None):
    """
    Evaluates the classifier using K-Fold cross-validation.

    Parameters
    ----------
    features : array-like
        Array of features
    labels : array-like
        Array of labels
    clf : object
        Classifier with methods fit and predict
    score_function : func, optional
        Function that receives y_true and y_pred and returns a score
    n_splits : int, optional
        Number of folds. Must be at least 2
    random_state : int, optional
        Controls the randomness of the fold assignment

    Returns
    -------
    average_score : float
        Average score of the classifier across all folds
    """
    logging.info("K-Fold accuracy\nClassifier:{}\nn_splits:{}\nscore_metric:{}".format(
        clf, n_splits, score_function))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_index, test_index in kf.split(features):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        score = score_function(labels_test, predictions)
        scores.append(score)

        logging.info("Fold score: {:.3f}".format(score))

    average_score = np.mean(scores)
    logging.info("Average K-Fold Score: {:.3f}".format(average_score))

    return average_score


def kfold_stratified_score(features, labels, clf, n_splits=4, score_function=balanced_accuracy_score,
                           random_state=None):
    """
    Computes KFold cross validation score using n_splits folds.
    It uses the features and labels to train the k-folds.
    Uses a stratified KFold split.
    The score_function is the one used to score each k-fold.

    Parameters
    ----------
    features : list
        List of features
    labels : list
        List of labels
    clf : object
        classifier with methods fit, predict and score
    n_splits : int
        number of splits
    score_function : func
        function that receives X, y and return a score
    random_state : int
        random state for the stratified kfold

    Returns
    -------
    mean_accuracy : float
        mean score among k-folds test splits
    """
    logging.info("KFold Stratified accuracy\nClassifier:{}\nn_splits:{}\n"
                 "score_metric:{}".format(clf, n_splits, score_function))

    clfs = []

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Split the dataset. The skf saves splits index
    skf.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)
    labels_predicted = labels.copy()

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    # Iterate over the KFolds and do stuff
    gen = skf.split(features, labels)
    for train_index, test_index in gen:
        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train the classifier
        t1 = time.process_time()
        logging.info("Training clf...")
        clf.fit(features_train, labels_train)
        t2 = time.process_time()
        logging.info("Processing time: {:.3f}".format(t2 - t1))

        # Labels predicted for test split
        labels_pred_test = clf.predict(features_test).reshape(-1)
        labels_predicted[test_index] = labels_pred_test

        accuracies_kfold.append(score_function(labels_test, labels_pred_test))

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)

        kcounter += 1
        clfs.append(clf)

    mean_accuracy = np.mean(accuracies_kfold)
    std = np.std(accuracies_kfold)
    logging.info("Accuracy: {:.3f} +- {:.3f}".format(round(mean_accuracy, 3), round(std, 3)))

    # return mean_accuracy, std, labels, labels_predicted, clfs
    return mean_accuracy


def temporal_kfold_score(features, labels, clf, n_splits=4, score_function=balanced_accuracy_score):
    """
    Computes KFold cross validation score using n_splits folds.
    It uses the features and labels to train the k-folds.
    Uses a temporal KFold split.
    The score_function is the one used to score each k-fold.

    Parameters
    ----------
    features : list
        List of features
    labels : list
        List of labels
    clf : object
        classifier with methods fit, predict and score
    n_splits : int
        number of splits
    score_function : func
        function that receives X, y and return a score

    Returns
    -------
    mean_accuracy : float
        mean score among k-folds test splits
    """
    logging.info("TemporalKFold accuracy\nClassifier:{}\nn_splits:{}\n"
                 "score_metric:{}".format(clf, n_splits, score_function))
    print("TemporalKFold accuracy\nClassifier:{}\nn_splits:{}\n"
          "score_metric:{}".format(clf, n_splits, score_function))

    clfs = []

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # Split the dataset. The skf saves splits index
    tscv.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)
    labels_predicted = labels.copy()

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    size_k = int(features.shape[0] / n_splits)
    # Iterate over the KFolds and do stuff
    gen = tscv.split(features, labels)
    for train_index, test_index in gen:
        train_index = train_index[:-size_k]
        if train_index.shape[0] != 0:
            # Splits
            features_train, features_test = features[train_index], features[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # Train the classifier
            t1 = time.process_time()
            logging.info("Training clf...")
            print("Training clf...")
            clf.fit(features_train, labels_train)
            t2 = time.process_time()
            logging.info("Processing time: {:.3f}".format(t2 - t1))
            print("Processing time: {:.3f}".format(t2 - t1))
            # Labels predicted for test split
            labels_pred_test = clf.predict(features_test)
            labels_predicted[test_index] = labels_pred_test

            accuracies_kfold.append(score_function(labels_test, labels_pred_test))

            labels_kfold.extend(labels_test)
            labels_kfold_predicted.extend(labels_pred_test)

            kcounter += 1
            clfs.append(clf)

    mean_accuracy = np.mean(accuracies_kfold)
    std = np.std(accuracies_kfold)
    logging.info("Accuracy: {:.2f} +- {:.2f}".format(round(mean_accuracy, 3), round(std, 3)))
    print("Accuracy: {:.2f} +- {:.2f}".format(round(mean_accuracy, 3), round(std, 3)))

    # return mean_accuracy, std, labels, labels_predicted, clfs
    return mean_accuracy
