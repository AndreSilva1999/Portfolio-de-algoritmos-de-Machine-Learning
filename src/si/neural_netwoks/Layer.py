import numpy as np
from si.statistics.sigmoid_function import sigmoid
from si.data.dataset1 import Dataset

class Dense:

    def __init__(self,input_size: int,output_size: int) -> None:
        """_summary_
        Initialize parameters and atributes
        Args:
            input_size (int): linhas
            output_size(int): colunas
        """
        #parameters
        self.input_size= input_size #Numero de linhas 
        self.output_size= output_size #Features #Numero de colunas

        #atributes
        self.weights= np.random.randn(input_size,output_size)*0.01 #Numero de linhas, numero de colunas respetivamente
        #Podemos acrescentar 0.01 para ajustar valores grandes!
        self.bias= np.zeros((1,output_size))
        

    def foward(self,input_data:np.ndarray)-> np.ndarray:
        """_summary_

        Args:
            input_data (np.ndarray)

        Returns:
            np.ndarray
        """
        #Nr de colunas da 1 matriz tem que ser igual ao numero das linhas da 2
        #Numero de input tem que ser igual ao numero de features
        return np.dot(input_data,self.weights) + self.bias

    def backwards(self, error:np.ndarray, learning_rate:float)-> np.ndarray:
        """_summary_

        Args:
            error (np.ndarray): _description_
            learning_rate (float): _description_

        Returns:
            np.ndarray: _description_
        """
        #Update weights and bias
        self.weights-= learning_rate*np.dot(self.X.T,error)

class SigmoidActivation:

    def __init__(self) -> None:
        pass

    def foward(self,input_data): 
        return sigmoid(input_data)


class SoftMaxActivation:
    """_summary_
    Generats a foward based on SoftMaxActivation function
    """
    def __init__(self) -> None:
        pass

    def foward(self, input_data:Dataset):
        """_summary_
        Foward
        Args:
            input_data (Dataset): Dataset

        Returns:
            _type_: Returns an np.ndarray
        """
        return (np.exp(max(input_data.X))/np.sum(np.exp(input_data.X,axis=1,keepdims=True)))


class ReLUActivation:
    """_summary_
     Generats a foward based on ReLUActivation function
    """
    def __init__(self) -> None:
        pass

    def foward(self,input_data:Dataset):
        """_summary_
        Foward
        Args:
            input_data (Dataset): Dataset

        Returns:
            _type_: Returns an np.ndarray
        """
        
        return np.maximum(min(input_data.X),input_data.X)