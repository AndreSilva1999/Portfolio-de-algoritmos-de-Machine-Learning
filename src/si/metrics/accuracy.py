import numpy as np

def accuracy(y_true:np.array, y_pred:np.array):
    """
    Describe
    """
    return np.sum((y_true==y_pred)/len(y_true))