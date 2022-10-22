<<<<<<< HEAD

import pandas as pd
import numpy as np
import pandas as pd
import sys
from si.f_classification import f_classification
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset


class SelectKBest:

    def __init__(self,score_func,k) -> None:
        self.score_func=f_classification(dataset)
        self.k=k
        self.F=None
        self.p=None

    def fit(self,dataset):
        self.f,self.p= self.score_func
        return self

    def transform(self,dataset)-> Dataset:
        idxs= np.argsort(self.F)[-self.k:]
        features= np.array(dataset.features)[idxs]
        return Dataset(dataset.x[:, idxs], dataset.y,list(features),dataset.label)

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)


if __name__== "__main__":
    # from read_csv import read_csv
    # data= read_csv("iris.csv",sep=",")
    # percent= selecetPercentile(data)
    # print(percent.fit_transform())
    x= np.array([[1,2,3],[1,2,3]])
    y= np.array([1,2])
    features=["A","B","C"]
    label= "y"
    dataset= Dataset(x=x,y=y,features=features,label=label)
    percent= SelectKBest(dataset,2)
=======

import pandas as pd
import numpy as np
import pandas as pd
import sys
from si.f_classification import f_classification
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset


class SelectKBest:

    def __init__(self,score_func,k) -> None:
        self.score_func=f_classification(dataset)
        self.k=k
        self.F=None
        self.p=None

    def fit(self,dataset):
        self.f,self.p= self.score_func
        return self

    def transform(self,dataset)-> Dataset:
        idxs= np.argsort(self.F)[-self.k:]
        features= np.array(dataset.features)[idxs]
        return Dataset(dataset.x[:, idxs], dataset.y,list(features),dataset.label)

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)


if __name__== "__main__":
    # from read_csv import read_csv
    # data= read_csv("iris.csv",sep=",")
    # percent= selecetPercentile(data)
    # print(percent.fit_transform())
    x= np.array([[1,2,3],[1,2,3]])
    y= np.array([1,2])
    features=["A","B","C"]
    label= "y"
    dataset= Dataset(x=x,y=y,features=features,label=label)
    percent= SelectKBest(dataset,2)
>>>>>>> 01e8830462667d8cacc40df23ee43f9e9137d949
    percent.fit_transform(dataset)