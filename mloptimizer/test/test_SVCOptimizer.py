import os
import shutil

import pytest
from sklearn.datasets import load_iris, load_breast_cancer

from mloptimizer.genoptimizer import Param
from mloptimizer.genoptimizer import SVCOptimizer