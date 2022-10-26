from itertools import permutations
import random
import pandas as pd
import numpy as np
import sys
from si.statistics.euclidean_distance import euclidean_distance
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from typing import Callable
from si.model_selection.split import train_test_split
from si.metrics.accuracy import accuracy

class KNNClassifier:

    def __init__(self, examples) -> None:
        self.k= examples
        self.distance= euclidean_distance
        self.dataset= None

    def fit(self,dataset,size):
        self.dataset= train_test_split(dataset,size)[0]
        return self

    def _get_nearest_label(self,sample:np.ndarray):
        #Obtem a distancia de uma sample ao dataset e retorna os k labels com maior proximidade
        distances= self.distance(sample,self.dataset.x)
        k_nearest= np.argsort(distances)[:self.k]
        k_nearest_labels= self.dataset.y[k_nearest,]
        labels,counts=np.unique(k_nearest_labels,return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self,dataset: Dataset):
        #Calcula a distância entre cada amostra e as várias amostras do dataset de treino
        return np.apply_along_axis(self._get_nearest_label,axis=1,arr= dataset.x)

    def score(self,dataset:Dataset):
        predict=self.predict(dataset)
        return accuracy(dataset.y,predict)



if __name__== "__main__":
    from src.si.read_csv import read_csv
    dataset_= read_csv("iris.csv",sep=",")
    print(dataset_.label)
    knn= KNNClassifier(3)
    knn.fit(dataset_,0.2)
    score= knn.score(dataset_)
    print(f'The accuracy of the model is: {score}')



    


