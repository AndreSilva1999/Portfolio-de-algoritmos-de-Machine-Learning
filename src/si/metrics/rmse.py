from itertools import permutations
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from typing import Callable
import math


def rmse(y_true:np.array, y_pred:np.array):
    """
    Get rmse(Root Mean Square Error)
    """

    MSE = np.square(np.subtract(y_true,y_pred)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE
