from contextlib import nullcontext
from formatter import NullFormatter
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/André Silva/SI/")
from si.data.dataset1 import Dataset
from typing import Callable


class PCA:

    def __init__(self, n:int) -> None:
        self.n_componentes= n
        self.mean= None
        self.componentes:np.ndarray= None
        self.explained_variance= None


    def _center_data(self,dataset:Dataset):
        #Cetra dados iniciais
        mean=dataset.get_average()
        center= np.subtract(dataset.x,mean)
        return center

    def _get_SVD(self,dataset: Dataset):
        #Calcula SVD
        U,S,Vt= np.linalg.svd(dataset.x,full_matrices=False)
        X= np.matmul(np.matmul(U, np.diag(S)), Vt)
        return X,Vt

    def fit(self,dataset):
        center = self._center_data(dataset)
        self.mean,Vt= self._get_SVD(dataset)
        #estima a média, os componentes e a variância explicada
        self.componentes= Vt[self.n_componentes-1]
        EV= self.mean/(center-1)
        self.explained_variance= EV[self.n_componentes-1]
        return self


    def transform(self,dataset: Dataset):
        #calcula o dataset reduzido usando os componentes principais
        self.fit(dataset)
        #Subtrai a media ao dataset
        print(self.componentes)
        center=np.subtract(dataset.x,self.componentes.T)
        #Reduçao....
        reduce_div= np.dot(center,self.componentes.T)
        return reduce_div



if __name__== "__main__":
    # x= np.array([[4,3,1],[2,6,2]])
    # y= np.array([1,2])
    # features=["A","B","C"]
    # label= "y"
    # data= Dataset(x=x,y=y,features=features,label=label)
    from si.read_csv import read_csv
    dataset_= read_csv("iris.csv",sep=",")        
    result= PCA(3)
    print(result.transform(dataset_))











