<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset


def f_classification(dataset:Dataset):
    classes = dataset.get_classes()
    groups = [dataset.x[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
<<<<<<< HEAD
=======
=======
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset


def f_classification(dataset:Dataset):
    classes = dataset.get_classes()
    groups = [dataset.x[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    print(F,p)
>>>>>>> 01e8830462667d8cacc40df23ee43f9e9137d949
>>>>>>> 640a6ab55fcf35ea9928cdd8a3af9f49bad5e9e8
    return F, p