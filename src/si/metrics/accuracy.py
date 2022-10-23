from itertools import permutations
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/André Silva/SI/")


def accuracy(y_true:np.array, y_pred:np.array):
    """
    Describe
    """
    return np.sum((y_true==y_pred)/len(y_true))