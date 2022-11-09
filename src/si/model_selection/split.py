from itertools import permutations
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/André Silva/SI/")
from si.data.dataset1 import Dataset
from typing import Callable


def train_test_split(dataset: Dataset, test_size: float, random_state:int = 42):
    data= dataset.shape()[0]
    np.random.seed(random_state) #Para obter os meus resultados com os mesmos dados
    test_size=test_size
    n_test=int(data*test_size)
    perm= np.random.permutation(data)
    test_indxs= perm[:n_test]
    train_indxs= perm[n_test:]
    train=Dataset(dataset.X[train_indxs], dataset.y[train_indxs], features=dataset.features, label=dataset.label)
    test= Dataset(dataset.X[test_indxs],dataset.y[test_indxs], features=dataset.features, label=dataset.label)
    return train,test

