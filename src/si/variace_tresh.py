import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset

class VarianceTreshold:

    def __init__(self,treshold) -> None:
        self.treshold= treshold
        self.variance= None
    

    def fit(self,dataset):
        variance= Dataset.get_variance(dataset)
        self.variance=variance
        return self

    def transform(self,dataset) -> Dataset:
        mask= self.variance > self.treshold
        newX=dataset.x[:, mask]
        print(mask)
        features=np.array(dataset.features)[mask]
        return Dataset(newX,dataset.y,features=list(features),label=dataset.label)
        

    def fit_transform(self,dataset):
        self.fit(dataset)
        get_new_set= self.transform(dataset)
        return get_new_set.summary()



if __name__== "__main__":
    x= np.array([[4,3,1],[2,6,2]])
    y= np.array([1,2])
    features=["A","B","C"]
    label= "y"
    data= Dataset(x=x,y=y,features=features,label=label)
    # print(data.summary())
    vars= VarianceTreshold(2)
    print(vars.fit_transform(data))