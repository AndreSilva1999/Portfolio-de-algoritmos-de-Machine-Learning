from itertools import permutations
import pandas as pd
import numpy as np
import sys
from si.statistics.f_classification import f_classification
from si.statistics.euclidean_distance import euclidean_distance
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from typing import Callable


class KNNClassifier:

    def __init__(self, examples,dataset:Dataset) -> None:
        self.k= examples
        self.distance= euclidean_distance
        self.data= None
        self.dataset= dataset

    def fit(self,dataset):
        self.data= dataset
        return self
    def _get_nearest_label(self,sample: np.ndarray):

        distances= self.distance(sample,self.dataset.x)
        k_nearest= np.argsort(distances)[:self.k]
        k_nearest_labels= self.dataset.y[k_nearest]
        labels,counts=np.unique
        return np.argmax(labels)

    def predict(self,dataset,sample:np.ndarray):

        return np.apply_along_axis(self._get_nearest_label,axis=1,arr= dataset.x)


