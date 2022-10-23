from itertools import permutations
import pandas as pd
import numpy as np
import sys
<<<<<<< HEAD
sys.path.append("/Users/André Silva/SI/")
=======
from si.f_classification import f_classification
from si.euclidean_distance import euclidean_distance
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from typing import Callable
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8


def accuracy(y_true:np.array, y_pred:np.array):
    """
    Describe
    """
    return np.sum((y_true==y_pred)/len(y_true))