"""
Quickstart example
====================================
Quick example of use of the library to optimize a decision tree classifier.
Firstly, we import the necessary libraries to get data and plot the results.
"""

from mloptimizer.application import Optimizer
from mloptimizer.domain.hyperspace import HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

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
hyperparam_space = HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier)

# %%
# The TreeOptimizer class is the main wrapper for the optimization of a Decision Tree Classifier.
# The first argument is the vector of features,
# the second is the vector of labels and
# the third (if provided) is the name of the folder where the results of mloptimizer Optimizers are saved.
# The default value for this folder is "Optimizer"
opt = Optimizer(estimator_class=DecisionTreeClassifier, features=X_train, labels=y_train,
                hyperparam_space=hyperparam_space, folder="Optimizer")

# %%
# To optimizer the classifier we need to call the optimize_clf method.
# The first argument is the number of generations and
# the second is the number of individuals in each generation.
# The method returns the best classifier with the best hyperparameters found.
clf = opt.optimize_clf(10, 10)

print(clf)

# %%
# Train the classifier with the best hyperparameters found
# Show the classification report and the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=clf.classes_,
    cmap=plt.cm.Blues
)
disp.plot()
plt.show()

del opt
