from si.data.dataset1 import Dataset
import numpy as np
import pandas as pd
from si.model_selection import split
from si.metrics.accuracy import accuracy

def cross_validate(model,dataset:Dataset,scoring,cv,test_size,random_state:int = 42):
    model=model
    dataset=dataset
    scoring=scoring
    cv=cv
    test_size=test_size
    scores={"seeds":[],
    "train":[],
    "test":[]
    }

    for i in range(cv):
         seed=np.random.randint(0,1000)
         scores["seeds"] += [seed]
         train, test= split(dataset,test_size,random_state=seed)
         model.fit(train)
    if scores is None:
        scores["train"].append(model.score(train))
        scores["test"].append(model.score(test))
    else:
        train= model.score(scoring,train)
        test= model.score(scoring,test)
        scores["train"]= train
        scores["test"]= test
