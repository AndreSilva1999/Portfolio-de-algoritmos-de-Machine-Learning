from itertools import permutations
import pandas as pd
import numpy as np
import sys
from si.f_classification import f_classification
from si.euclidean_distance import euclidean_distance
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from typing import Callable


def train_test_split(dataset: Dataset, testsize: float, random_state:int = 42):
    data= dataset.shape()[0]
    np.random.seed(random_state)
    n_test=int(data*test_size)
    perm= np.random.permutation(data)
    test_size= perm[:n_test]
    train_indxs= permutations[n_test:]

    train=Dataset(dataset.x[train_indxs])
    test= Dataset(dataset.x[test_size],y=dataset.y, features=dataset.features, label= dataset.label)
    return train,test

