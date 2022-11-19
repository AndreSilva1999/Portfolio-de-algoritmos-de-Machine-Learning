
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset


def f_classification(dataset:Dataset):
    """_summary_

    Args:
        dataset (Dataset): Dataset

    Returns:
        f,p : Returns the freq and prob of each 
    """
    classes = dataset.get_classes()
    groups = [dataset.x[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p