
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
        """_summary_
        Helper fucntion to get Full combinations
        Uses itertoools to produce the product of size k
        """
        alphabete= ["A","T","G","C"]
        combinations=itertools.product(alphabete,repeat=self.k)
        return("%s"*self.k % tup for tup in (combinations))
    
    def _get_kmers(self,sequence):
        """_summary_

        Args:
            sequence (list): return a list within a list [["ACTGACATCTACT"]]
            IMPORTANT! USE [0] TO GET STRING!

        Returns:
            KMERs(np.array): Return a np.array with all kmers of all sequences [[kmer1],[kmer2],[kmer3]....]
        """
        return np.array((sequence[0][i:i+self.k]) for i in range(len(sequence[0])-self.k+1))
    
    def fit(self,dataset):
        """_summary_
        Does the fit
        Args:
            dataset (DATASET): Gets a dataset of nucleotids

        Returns:
            returns self

        """
        print(dataset.X[0])
        self.kmers=np.apply_along_axis(self._get_kmers,axis=1,arr=dataset.X)
        self.fite= True
        return self

    def _get_frq(self,seq):

        """_summary_
        Gets the counts for each count that appears in the sequence
        Args:
            sequence (list): return a list within a list [["ACTGACATCTACT"]]
            IMPORTANT! USE [0] TO GET STRING!
        """
        combinations= self._get_all_combinations()
        counts= {kmer:0 for kmer in combinations}
        for value in self.kmers[self.counts]:
            counts[value]+=1
        self.counts+=1
        return(np.array([counts[i]/len(seq[0]) for i in counts]))


    def transform(self,dataset: Dataset):        
        """_summary_
        Transforms the dataset to get an informative value of the data
        Args:
            dataset(Dataset): Nucleotide dataset
        """
        if self.fite:
            freq= np.apply_along_axis(self._get_frq,axis=1,arr=dataset.X)
            return Dataset(freq,y=dataset.y,features=dataset.features,label= dataset.label)
        else:
            raise("Please fit data first")

    def fit_transform(self,dataset: Dataset):
        """_summary_
        Does fit and transform
        Args:
            dataset(Dataset): Nucleotide dataset

        Returns:
            _type_: Return the new dataset 
        """
        self.fit(dataset)
        new_data=self.transform(dataset)
        return new_data



if __name__== "__main__":
    from si.read_csv import read_csv
    dataset= read_csv("/Users/André Silva/SI/datasets/tfbs.csv")
    kmer= KMer(3)
    new_data= kmer.fit_transform(dataset)
    print(new_data.X)
    