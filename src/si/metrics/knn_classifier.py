
import numpy as np
import sys
from si.statistics.euclidean_distance import euclidean_distance
sys.path.append("/Users/André Silva/SI/")
from dataset import Dataset
from si.metrics.accuracy import accuracy

class KNNClassifier:

    def __init__(self, examples: float,dataset:Dataset) -> None:
        """_summary_
        Initialize parameters
        Args:
            examples (float): _description_
            dataset (Dataset): _description_
        """
        self.k= examples
        self.distance= euclidean_distance
        self.data= None
        self.dataset= dataset

    def fit(self,dataset):
        """_summary_

        Args:
            dataset (Dataset): Dataset

        Returns:
            Self
        """
        self.data= dataset
        return self
    def _get_nearest_label(self,sample: np.ndarray):
        """_summary_
        Get nearest k neightbors
        Args:
            sample (np.ndarray): Sample of X

        Returns:
            _type_: _description_
        """
        distances= self.distance(sample,self.dataset.x)
        k_nearest= np.argsort(distances)[:self.k]
        k_nearest_labels= self.dataset.y[k_nearest]
        labels,counts=np.unique
        return np.argmax(labels)

    def predict(self,dataset,sample:np.ndarray):
        """_summary_
        Does predict
        Args:
            dataset (_type_): _description_
            sample (np.ndarray): _description_

        Returns:
            Np.array
        """
        return np.apply_along_axis(self._get_nearest_label,axis=1,arr= dataset.x)

    def score(self,dataset:Dataset):
        """_summary_

        Args:
            dataset (Dataset): Dataset

        Returns:
           Scores
        """
        predict=self.predict(dataset)
        return accuracy(dataset.y,predict)
