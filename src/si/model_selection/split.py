from itertools import permutations
import pandas as pd
import numpy as np
import sys
<<<<<<< HEAD
=======
from si.f_classification import f_classification
from si.euclidean_distance import euclidean_distance
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from typing import Callable


<<<<<<< HEAD
def train_test_split(dataset: Dataset, test_size: float, random_state:int = 42):
    data= dataset.shape()[0]
    np.random.seed(random_state) #Para obter os meus resultados com os mesmos dados
    test_size=test_size
    n_test=int(data*test_size)
    perm= np.random.permutation(data)
    test_indxs= perm[:n_test]
    train_indxs= perm[n_test:]

    train=Dataset(dataset.x[train_indxs], dataset.y[train_indxs], features=dataset.features, label=dataset.label)
    test= Dataset(dataset.x[test_indxs],dataset.y[test_indxs], features=dataset.features, label=dataset.label)
=======
def train_test_split(dataset: Dataset, testsize: float, random_state:int = 42):
    data= dataset.shape()[0]
    np.random.seed(random_state)
    n_test=int(data*test_size)
    perm= np.random.permutation(data)
    test_size= perm[:n_test]
    train_indxs= permutations[n_test:]

    train=Dataset(dataset.x[train_indxs])
    test= Dataset(dataset.x[test_size],y=dataset.y, features=dataset.features, label= dataset.label)
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
    return train,test

