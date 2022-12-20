
from si.data.dataset1 import Dataset
import numpy as np
import pandas as pd
import itertools
from si.model_selection.cross_validate import cross_validate

def grid_search_cv(model,dataset:Dataset,cv,parameter_grid: dict,test_size = 0.2,scoring=None):
    model=model

    # validate the parameter grid

    for parameter in parameter_grid:
        if not hasattr(model,parameter):
            raise AttributeError(f"Model{model} does not have parameter{parameter}.")
        
    scores=[]
        
        # for each combination

    for combination in itertools.product(*parameter_grid.values()):
        parameters={}
        # parameter configuration
        for parameter,value in zip(parameter_grid.keys(),combination):
            setattr(model,parameter,value)
            parameters[parameter]=value
        # cross validate the model

        score= cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv,test_size= test_size)
        # add the parameter configuration
        score["parameters"]=parameters
        # add the score
        scores.append(score)
    return scores



if __name__ == '__main__':
    # import dataset
    from si.data.dataset1 import Dataset
    from si.linear_model.logistic_regression import LogisticRegression
    from si.read_csv import read_csv
    from sklearn.preprocessing import StandardScaler

    # load and split the dataset
    breast_dataset = read_csv("/Users/André Silva/SI/datasets/breast-bin.csv")
    breast_dataset.X = StandardScaler().fit_transform(breast_dataset.X)

    # initialize the Logistic Regression model
    logs = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = grid_search_cv(logs,
                             breast_dataset,
                             parameter_grid=parameter_grid_,
                             cv=3)

    # print the scores
    print(scores_)