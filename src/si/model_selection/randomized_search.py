from si.data.dataset1 import Dataset
import numpy as np
import pandas as pd
import itertools
from si.model_selection.cross_validate import cross_validate



def rand_search(model,dataset:Dataset,parameter_distribution: dict,n_iter: int,test_size: int=0.2,scoring=None,cv: int=3) -> None:
    """
    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    test_size: float
        The test size.
    parameter_distribution: dict
        Dictionary of each parameter 
    """

     # validate the parameter distribution

    for parameter in parameter_distribution:
        if not hasattr(model,parameter):
            raise AttributeError(f"Model{model} does not have parameter{parameter}.")


    scores=[]
    #Get an random combination of set parameters n_iter times
    for _ in range(n_iter):
        parameters={}
        # parameter configuration

        for parameter in parameter_distribution:
            value= np.random.choice(parameter_distribution[parameter])
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

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_des_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = rand_search(knn,dataset_,parameter_distribution=parameter_des_,n_iter=10,cv=3)

    # print the scores
    print(scores_)

