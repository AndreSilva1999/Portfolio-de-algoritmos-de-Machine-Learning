from si.data.dataset1 import Dataset
import numpy as np
import pandas as pd
from si.model_selection import split
from si.metrics.accuracy import accuracy
import itertools
from si.model_selection import cross_validate

def grid_search_cv(model,dataset:Dataset,test_size,scoring,cv,parameter_grid: dict):
    model=model

    for parameter in parameter_grid:
        if not hasattr(model,parameter):
            raise AttributeError(f"Model{model} does not have parameter{parameter}.")
        
        scores=[]

        for combination in itertools.products(*parameter_grid.values):
            parameters={}

            for parameter,value in zip(parameter_grid.keys(),combination):
                setattr(model,parameter,value)
                parameters[parameter]=value

            score= cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv,test_size= test_size)

            score["parameters"]=parameters

            scores.append(score)




