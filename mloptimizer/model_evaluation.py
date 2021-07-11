import logging
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import numpy as np
import time


def unpackDF(df, target_variable="class"):
    '''
    Extract classes, features, and labels from a pandas.DataFrame.
    One column of the DataFrame should be called "class" and
    the rest are features.

    :param DataFrame df: pandas.DataFrame with the dataset
    :return: Classes, features, labels
    '''
    class_list = list(df[target_variable].drop_duplicates())
    labels = df[target_variable]
    features = np.array(df.drop(columns=[target_variable]))
    return class_list, features, labels


def KFoldStratifiedAccuracy(features, labels, clf, n_splits=4, score_function=balanced_accuracy_score,
                            random_state=None):
    '''
    Computes KFold cross validation accuracy using n_splits folds over the data in the pandas.DataFrame given.
    Uses an stratified KFold with the random_state specified.

    :param df: pandas.DataFrame where is the data for train/test splits
    :param clf: classifier with methods fit, predict and score
    :param n_splits: number of splits
    :param random_state: random state seed
    :return: mean accuracy, std
    '''
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

    meanAccuracy = np.mean(accuracies_kfold)
    std = np.std(accuracies_kfold)
    logging.info("Accuracy: {:.2f} +- {:.2f}".format(round(meanAccuracy, 3), round(std, 3)))

    # return meanAccuracy, std, labels, labels_predicted, clfs
    return meanAccuracy, std


def TemporalKFoldAccuracy(features, labels, clf, n_splits=4, score_function=balanced_accuracy_score):
    '''
    Computes KFold cross validation accuracy using n_splits folds over the data in the pandas.DataFrame given.
    Uses an stratified KFold with the random_state specified.

    :param df: pandas.DataFrame where is the data for train/test splits
    :param clf: classifier with methods fit, predict and score
    :param n_splits: number of splits
    :param random_state: random state seed
    :return: mean accuracy, std
    '''
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

    meanAccuracy = np.mean(accuracies_kfold)
    std = np.std(accuracies_kfold)
    logging.info("Accuracy: {:.2f} +- {:.2f}".format(round(meanAccuracy, 3), round(std, 3)))
    print("Accuracy: {:.2f} +- {:.2f}".format(round(meanAccuracy, 3), round(std, 3)))

    # return meanAccuracy, std, labels, labels_predicted, clfs
    return meanAccuracy, std
