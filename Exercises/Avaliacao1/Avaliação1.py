# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class Dataset:

    def __init__(self,name,sep=";") -> None:
        self.data= pd.read_csv(name,sep)
    
    def print_data(self):
        return self.data
    
    def give_first(self):
        return self.data.iloc[:,0]
    
    def give_first_x_mean(self,x):
        last_five=self.data.loc[:x-1].mean()
        return last_five
        
    def more_than(self,x):
       more= self.data[self.data.iloc[:,0:3]>=x]
       return more

    def get_class(self,name):
        return self.data[self.data["class"]== f"{name}"]
        
    def dropna(self,sub=False):
        if not sub:
            droped= self.subNA()
            return droped
        else:
            droped= self.data.dropna(how="all")
            return droped
    
    def subNA(self):
        self.data["class"].fillna("No_class",inplace=True)
        self.data.fillna(0,inplace=True)
        return self.data
    
if __name__=="__main__":
    dataset= Dataset("iris.csv",sep=",")
    #print(dataset.print_data())
    # print(dataset.give_first())
    # print(dataset.give_first_x_mean(5))
    # print(dataset.more_than(1))
    # print(dataset.get_class("Iris-setosa"))
    print(dataset.dropna(True))