Introduction
==================

`mloptimizer` is a Python library developed to enhance the performance of machine learning models through the optimization of hyperparameters using genetic algorithms. By applying principles inspired by natural selection, genetic algorithms allow `mloptimizer` to explore large hyperparameter spaces efficiently. This approach not only reduces search time and energy consumption but also achieves results comparable to more computationally demanding search methods.

This User Guide provides a comprehensive overview of `mloptimizer`’s functionality, setup, and usage. It’s designed for users who are familiar with machine learning libraries like :mod:`scikit-learn` and are looking to incorporate more flexible optimization techniques into their workflows.

The library’s syntax is intentionally similar to :class:`scikit-learn <sklearn.model_selection.GridSearchCV>`'s hyperparameter search tools, but with added layers of customization and control. `mloptimizer` supports a range of popular algorithms, including :class:`DecisionTreeClassifier <sklearn.tree.DecisionTreeClassifier>`, :class:`RandomForestClassifier <sklearn.ensemble.RandomForestClassifier>`, and :class:`XGBClassifier <xgboot.XGBClassifier>`. Additionally, it is designed to be compatible with other models that follow the :class:`Estimator <sklearn.base.Estimator>` class from the :mod:`scikit-learn` API, allowing for easy integration into existing projects.

This guide will walk you through everything from installation and quickstart examples to more advanced concepts, customization options, and visualization tools. While `mloptimizer` is designed to be user-friendly, it also offers advanced configuration options for users seeking fine-grained control over their optimization processes. The guide reflects the current functionality of `mloptimizer` and will be updated as the library evolves.

.. toctree::
   :hidden:

   features
   overview

