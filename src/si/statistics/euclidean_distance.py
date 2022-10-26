import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset


def euclidean_distance(amostra=np.array, v_amostras=np.array):
    x=amostra
    y=v_amostras
    dist= np.sqrt(((x-y)**2).sum(axis=1))
    return dist
