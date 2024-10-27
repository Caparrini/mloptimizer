"""
Quickstart example
====================================
Quick example of use of the library to optimize a decision tree classifier.
Firstly, we import the necessary libraries to get data and plot the results.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch

# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# Define the HyperparameterSpace, you can use the default hyperparameters for the machine learning model
# that you want to optimize. In this case we use the default hyperparameters for a DecisionTreeClassifier.
# Another dataset or a custom one can be used
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(estimator_class=DecisionTreeClassifier)

# %%
# The GeneticSearch class is the main wrapper for the optimization of a machine learning model.
opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
    )

# %%
# To optimizer the classifier we need to call the optimize_clf method.
# The first argument is the number of generations and
# the second is the number of individuals in each generation.
# The method returns the best classifier with the best hyperparameters found.
opt.fit(X, y, 100, 30)

print(opt.best_estimator_)

# %%
# Train the classifier with the best hyperparameters found
# Show the classification report and the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = opt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=opt.best_estimator_.classes_,
    cmap=plt.cm.Blues
)
disp.plot()
plt.show()

del opt
