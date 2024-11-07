====================
Install
====================
This package is available on the `Python Package Index (PyPI)
<http://pypi.python.org/pypi/mloptimizer>`__, so the easiest way to install it is using ``pip``:

.. code:: bash

    pip install mloptimizer

.. warning::

        To avoid conflicts with other packages, it is recommended to install ``mloptimizer`` in a virtual
        environment. See the section below for more information.


Alternatively, you can find ``mloptimizer`` on `GitHub
<http://github.com/Caparrini/mloptimizer>`__ if you'd like to clone the repository and explore its code.

Virtual environment
-------------------

We recommend creating a virtual environment using Python's built-in ``venv`` package. To learn more, check the official
Python documentation at https://docs.python.org/3/library/venv.html.

Use the following commands to create and activate a virtual environment:

.. code:: bash

   # Create the virtual environment
   python -m venv myenv
   # Activate the virtual environment
   source myenv/bin/activate

Once the environment is activated, install ``mloptimizer`` by running:

.. code:: bash

   pip install mloptimizer

=====================
Verification
=====================

After installation, follow these steps to confirm that ``mloptimizer`` is correctly installed and functioning.

.. warning::

        Before running the tests, ensure you have the following packages installed, as ``mloptimizer`` does not include them:

        - ``pytest`` for running tests
        - ``pytest-cov`` for coverage analysis
        - ``pytest-mock`` for mocking functionality
        - ``mlflow`` for some of the test components

        You can install them by running:

        .. code:: bash

            pip install pytest pytest-cov pytest-mock mlflow

1. **Check the Installed Version**

   Run the following command to confirm the installed version of ``mloptimizer``:

   .. code:: bash

       python -m mloptimizer --version

   This should display the version of ``mloptimizer`` you’ve installed. If you see an error, the installation may need to be reviewed.

2. **Run Tests with `pytest`**

   To verify that all functionalities of ``mloptimizer`` are operational, execute:

   .. code:: bash

       pytest --pyargs mloptimizer

   This command runs all available tests to confirm the package is working correctly.

===============
Troubleshooting
===============

If you experience any issues during installation or testing of ``mloptimizer``, feel free to submit a report to the `issue tracker <https://github.com/Caparrini/mloptimizer/issues>`_. Before doing so, please review common issues and solutions.

Currently, there are no known installation issues.
