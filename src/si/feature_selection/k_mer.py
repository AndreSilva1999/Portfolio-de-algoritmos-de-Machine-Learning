
import numpy as np
import itertools
from si.data.dataset1 import Dataset

class KMer:

    def __init__(self,size: int=3) -> None:
        """_summary_
         Divide dna or rna into substrings of size "size"
        Args:
            size (int, optional): size of each substring 
        """
        self.k= size
        self.k_mers=None
        self.fite= False
        self.counts = 0

    def _get_all_combinations(self):
        alphabete= ["A","T","G","C"]
        combinations=itertools.product(alphabete,repeat=self.k)
        return("%s"*self.k % tup for tup in (combinations))
    
    def _get_kmers(self,sequence):
        return np.array((sequence[0][i:i+self.k]) for i in range(len(sequence[0])-self.k+1))
    
    def fit(self,dataset):
        print(dataset.X[0])
        self.kmers=np.apply_along_axis(self._get_kmers,axis=1,arr=dataset.X)
        self.fite= True
        return self

    def _get_frq(self,seq):
        combinations= self._get_all_combinations()
        counts= {kmer:0 for kmer in combinations}
        for value in self.kmers[self.counts]:
            counts[value]+=1
        self.counts+=1
        return(np.array([counts[i]/len(seq[0]) for i in counts]))


    def transform(self,dataset: Dataset):        
        if self.fite:
            freq= np.apply_along_axis(self._get_frq,axis=1,arr=dataset.X)
            return Dataset(freq,y=dataset.y,features=dataset.features,label= dataset.label)
        else:
            raise("Please fit data first")

    def fit_transform(self,dataset: Dataset):
        self.fit(dataset)
        new_data=self.transform(dataset)
        return new_data



if __name__== "__main__":
    from si.read_csv import read_csv
    dataset= read_csv("/Users/André Silva/SI/datasets/tfbs.csv")
    kmer= KMer(3)
    new_data= kmer.fit_transform(dataset)
    print(new_data.X)
    