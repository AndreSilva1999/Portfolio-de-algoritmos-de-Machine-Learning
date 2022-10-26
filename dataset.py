import pandas as pd
import numpy as np
from typing import Tuple, Sequence

class Dataset:

    def __init__(self,x,y,features,label) -> None:
        self.x=x #np.array
        self.y=y #np. 1d array
        self.features=features #lista de string
        self.label= label #string
    
    def shape(self):
        return self.x.shape

    def labels(self):
        if self.label:
            return True
        else:
            return False
        
    def get_classes(self):
        return np.unique(self.y)
        
    def get_average(self):
        return np.mean(self.x)
    
    def get_variance(self):
        return np.nanvar(self.x,axis=0)
    
    def get_median(self):
        return np.median(self.x) #também funciona assim
        
    def get_min_max(self):
        return np.amin(self.x), np.amax(self.x)
    
    def summary(self):
        exp_dataframe = pd.DataFrame(self.x,index=[self.y],
                                     columns=[self.features])
        return exp_dataframe
    
    def dropna(self):
        newdata= self.x[~np.isnan(self.x).any(axis=1)] # "~" refere se ao contrario, ou seja todas as linhas em que o nulo nao esteja
        index_list=[np.any(i) for i in np.isnan(self.x)]
        new_y=[]
        for i in range(len(index_list)):
           if index_list[i]:
                pass
           else:
               new_y.append(self.y[i])
        new_y=np.array(new_y)
        exp_dataframe = pd.DataFrame(newdata,index=[new_y],
                                     columns=[self.features], dtype = int)
        return exp_dataframe

    def fillna_zero(self): #
        newdata= np.nan_to_num(self.x)
        exp_dataframe = pd.DataFrame(newdata,index=[self.y],
                                     columns=[self.features], dtype = int)
        
        return exp_dataframe

    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        

        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)



if __name__=="__main__":
    # x= np.array([[1,2,3],[1,2,3]])
    # y= np.array([1,2])
    # features=["A","B","C"]
    # label= "y"
    # dataset= Dataset(x=x,y=y,features=features,label=label)
    # print(dataset.shape())   
    # print(dataset.labels())
    # print(dataset.get_classes())
    # # print(dataset.get_average())
    # # print(dataset.get_variance())
    # # print(dataset.get_median())
    # # print(dataset.get_min_max())
    # print(dataset.summary())
    dataset = Dataset.from_random(600, 100, 2)
    print(dataset)

=======
    
if __name__=="__main__":
    x= np.array([[1,2,3],[1,2,3]])
    y= np.array([1,2])
    features=["A","B","C"]
    label= "y"
    dataset= Dataset(x=x,y=y,features=features,label=label)
    # print(dataset.shape())   
    # print(dataset.labels())
    print(dataset.get_classes())
    # print(dataset.get_average())
    # print(dataset.get_variance())
    # print(dataset.get_median())
    # print(dataset.get_min_max())
    print(dataset.summary())
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
    #dataset.summary()
    # print(dataset.dropna())
        


