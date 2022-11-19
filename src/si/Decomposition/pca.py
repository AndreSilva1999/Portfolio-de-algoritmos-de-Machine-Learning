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
        """_summary_
        Puts dataset.X data in diferent groups
        Args:
            n (int): Number of components
        """

        self.n_componentes= n
        self.mean= None
        self.componentes:np.ndarray= None
        self.explained_variance= None


    def _center_data(self,dataset:Dataset):
        """_summary_

        Args:
            dataset (Dataset): Dataset

        Returns:
            Centered data
            Returns dataset.x centrado
        """
        #Centra dados iniciais
        mean=dataset.get_mean()
        center= np.subtract(dataset.X,mean)
        return center

    def _get_SVD(self,dataset: Dataset):
        """_summary_

        Args:
            dataset (Dataset): Dataset

        Returns:
            X,Vt
        """
        #Calcula SVD
        U,S,Vt= np.linalg.svd(dataset.X,full_matrices=False)
        X= np.matmul(np.matmul(U, np.diag(S)), Vt)
        return X,Vt

    def fit(self,dataset):
        """_summary_
        Does fit
        Args:
            dataset (Dataset): Dataset

        Returns:
            Returns components 
        """
        center = self._center_data(dataset)
        self.mean,Vt= self._get_SVD(dataset)
        #estima a média, os componentes e a variância explicada
        self.componentes= Vt[self.n_componentes-1]
        EV= self.mean/(center-1)
        self.explained_variance= EV[self.n_componentes-1]
        return self


    def transform(self,dataset: Dataset):
        """_summary_
        
        Args:
            dataset (Dataset): _description_

        Returns:
            _type_: _description_
        """
        #calcula o dataset reduzido usando os componentes principais
        self.fit(dataset)
        #Subtrai a media ao dataset
        print(self.componentes)
        center=np.subtract(dataset.X,self.componentes.T)
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











