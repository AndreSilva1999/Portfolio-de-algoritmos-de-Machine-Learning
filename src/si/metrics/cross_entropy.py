import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """_summary_
    Gives a score based on cross_entropy  
    Args:
        y_true (np.ndarray): True label
        y_pred (np.ndarray): Predicted label

    Returns:
        float: Returns a score
    """
    n= y_pred.shape[0]
    return -np.sum(y_true*np.log(y_pred))/n


def cross_entropy_derivate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """_summary_
    Gives a score based on cross entropy derivate
    Args:
        y_true (np.ndarray): True label
        y_pred (np.ndarray): Predicted labels

    Returns:
        float: Returns a score
    """

    n= y_pred.shape[0]
    return y_pred-y_true
    # return -y_true/(y_pred*n)
    