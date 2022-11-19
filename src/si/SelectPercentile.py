import pandas as pd
import numpy as np
import scipy as sp
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from si.statistics.f_classification import f_classification

class selecetPercentile:
    
    def __init__(self,dataset,p) -> None:
        """_summary_
        Initialize parameters
        Args:   
            dataset (Dataset): Dataset
            p (float): percentage
        """
        self.dataset=dataset
        self.score_func= f_classification(dataset)
        self.percentile= int(p*len(dataset.features))#Acrecentar o  tamanho do dataset (Eu quero p*features = 0.50*20=10)
        self.F=None
        self.p=None
    
    def fit(self):
        """_summary_
            Does fit
        Returns :
            Self
        """
        self.F, self.p= self.score_func
        return self

    def transform(self):
        """_summary_
        Select k best features
        Args:
            dataset (Dataset): Dataset

        Returns:
            Dataset: 
        """
        indxs= np.argsort(self.F)[-self.percentile]
        features= np.array(self.dataset.features)[indxs]
        return Dataset(self.dataset.x[:, indxs], self.dataset.y,list(features), self.dataset.label)

    def fit_transform(self):
        """_summary_
        Does fit and transform 
        Args:
            dataset (_type_): Dataset

        Returns:
            _type_: Returns full dataset
        """
        self.fit()
        return self.transform()



if __name__== "__main__":
    from si.read_csv import read_csv
    data= read_csv("iris.csv",sep=",")
    percent= selecetPercentile(data,0.50)
    # print(data.get_classes())
    percent.fit_transform()
    # x= np.array([[1,2,3],[1,2,3]])
    # y= np.array([1,2])
    # features=["A","B","C"]
    # label= "y"
    # dataset= Dataset(x=x,y=y,features=features,label=label)
    # percent= selecetPercentile(dataset,0.50)
    # print(percent.fit_transform())