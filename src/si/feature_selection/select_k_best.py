import pandas as pd
import numpy as np
import pandas as pd
import sys
from si.statistics.f_classification import f_classification

from si.data.dataset1 import Dataset


class SelectKBest:
    """_summary_
    Select K best features
    Uses our score_func to give the f and p of each feature
    Selects k best ones!
    """
    def __init__(self,k,dataset) -> None:
        """_summary_
        Initialize parameters
        Args:
            k (_type_): _description_
            dataset (_type_): _description_
        """
        self.score_func=f_classification(dataset)
        self.k=k
        self.F=None
        self.p=None

    def fit(self):
        """_summary_
            Does fit
        Returns :
            Self
        """
        self.F,self.p= self.score_func
        return self

    def transform(self,dataset: Dataset)-> Dataset:
        """_summary_
        Select k best features
        Args:
            dataset (Dataset): _description_

        Returns:
            Dataset: _description_
        """
        idxs= np.argsort(self.F)[-self.k:] #Ordena F da melhor a pior
        features= np.array(dataset.features)[idxs] #Seleciona indices das features 
        return Dataset(dataset.X[:, idxs], dataset.y,list(features),dataset.label)

    def fit_transform(self, dataset):
        """_summary_
        Does fit and transform 
        Args:
            dataset (_type_): Dataset

        Returns:
            _type_: Returns full dataset
        """
        self.fit()
        return self.transform(dataset)


if __name__== "__main__":
    from si.read_csv import read_csv
    data= read_csv("iris.csv",sep=",")
    percent= SelectKBest(3,dataset=data)
    newdata= percent.fit_transform(data)
    print(newdata.X)
    # x= np.array([[1,2,3],[1,2,3]])
    # y= np.array([1,2])
    # features=["A","B","C"]
    # label= "y"
    # dataset= Dataset(X=x,y=y,features=features,label=label)
    # percent= SelectKBest(2,dataset)
    # percent.fit_transform(dataset)