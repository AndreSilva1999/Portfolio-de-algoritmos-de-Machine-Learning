<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
import pandas as pd
import numpy as np
import scipy as sp
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from si.f_classification import f_classification

class selecetPercentile:
    
    def __init__(self,dataset,p) -> None:
        self.dataset=dataset
        self.score_func= f_classification(dataset)
        self.percentile= int(p*len(dataset.features))#Acrecentar o  tamanho do dataset
        self.F=None
        self.p=None
    
    def fit(self):
        self.F, self.p= self.score_func
        return self

    def transform(self):
        indxs= np.argsort(self.F)[-self.percentile]
        features= np.array(self.dataset.features)[indxs]
        return Dataset(self.dataset.x[:, indxs], self.dataset.y,list(features), self.dataset.label)

    def fit_transform(self):
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
<<<<<<< HEAD
=======
=======
import pandas as pd
import numpy as np
import scipy as sp
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from si.f_classification import f_classification

class selecetPercentile:
    
    def __init__(self,dataset,p) -> None:
        self.dataset=dataset
        self.score_func= f_classification(dataset)
        self.percentile= int(p*len(dataset.features))#Acrecentar o  tamanho do dataset
        self.F=None
        self.p=None
    
    def fit(self):
        self.F, self.p= self.score_func
        return self

    def transform(self):
        indxs= np.argsort(self.F)[-self.percentile]
        features= np.array(self.dataset.features)[indxs]
        return Dataset(self.dataset.x[:, indxs], self.dataset.y,list(features), self.dataset.label)

    def fit_transform(self):
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
>>>>>>> 01e8830462667d8cacc40df23ee43f9e9137d949
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
    # print(percent.fit_transform())