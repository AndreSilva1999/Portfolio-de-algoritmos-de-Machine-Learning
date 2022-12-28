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
        self.weights= np.random.randn(input_size,output_size) #Numero de linhas, numero de colunas respetivamente
        #Podemos acrescentar 0.01 para ajustar valores grandes!
        self.bias= np.zeros((1,output_size))
        self.X= None
        

    def forward(self,input_data:np.ndarray)-> np.ndarray:
        """Foward
        Args:
            input_data (np.ndarray)

        Returns:
            np.ndarray
        """
        #Nr de colunas da 1 matriz tem que ser igual ao numero das linhas da 2
        #Numero de input tem que ser igual ao numero de features
        self.X= input_data
        return np.dot(input_data,self.weights) + self.bias
        

    def backward(self, error:np.ndarray, learning_rate:float)-> np.ndarray:
        """backwards
        Args:
            error (np.ndarray): ndarray
            learning_rate (float): float

        Returns:
            np.ndarray: ndarray
        """
        #Get error of previous layer
        error_to_propagate= np.dot(error,self.weights.T)
        #Update weights and bias
        self.weights = self.weights - learning_rate*np.dot(self.X.T,error)
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)
        #Return error for previous layer
        return error_to_propagate



class SigmoidActivation:
    """_summary_
    Foward 
    """
    def __init__(self) -> None:
        self.X=None

    def forward(self,input_data: np.ndarray): 
        """_summary_

        Args:
            input_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.X= input_data
        sigmoid_data= sigmoid(input_data)
        return sigmoid_data

    def backward(self,error: np.ndarray,learning_rate):
        """_summary_

        Args:
            input_data (_type_): _description_
        """
        sigmoid_data= sigmoid(self.X)
        sigmoid_derivate=sigmoid_data*(1-sigmoid_data)
        error_to_propagate= error * sigmoid_derivate
        return error_to_propagate




class SoftMaxActivation:
    """_summary_
    Generats a foward based on SoftMaxActivation function
    """
    def __init__(self) -> None:
        self.X=None

    def forward(self, input_data:Dataset):
        """_summary_
        Foward
        Args:
            input_data (Dataset): Dataset

        Returns:
            _type_: Returns an np.ndarray
        """
        self.X= input_data

        z_exp = np.exp(input_data - np.amax(input_data))
        z_sum = np.sum(z_exp)
        return z_exp / z_sum

    def backward(self,error: float,learning_rate):
        """_summary_
        Backpropagation
        Args:
            input_data (Dataset): Dataset
            error (float): float
        """
        #We need to use the output of our layer, or foward to get the error 
        # error_to_prop= input_data * (error-np.sum(error*input_data,axis=1,keepdims=True))
        # return error_to_prop
        return error

    
class ReLUActivation:
    """_summary_
     Generats a foward based on ReLUActivation function
    """
    def __init__(self) -> None:
        self.X= None

    def forward(self,input_data:np.ndarray):
        """Foward
        Args:
            input_data (Dataset): Dataset

        Returns:
            _type_: Returns an np.ndarray
        """
        self.X= input_data
        return np.maximum(input_data,0)

    def backward(self,error:np.ndarray,learning_rate)-> np.ndarray:
        """backpropagation

        Args:
            input_data (Dataset): Dataset

        Returns: np.darray
        """

        self.X= np.where(self.X >1, 1,0)
        error_to_prop= error * (self.X)
        return error_to_prop


class LinearActivation():

    def __init__(self) -> None:
        self.X=None

    def forward(self, input_data: np.ndarray)-> np.ndarray:
        """Foward

        Args:
            input_data (np.ndarray): np.ndarray

        Returns:
            np.ndarray: np.ndarray
        """
        self.X= input_data
        return input_data
    
    def backward(self, error: np.ndarray, learning_rate)->np.ndarray:
        """Backward
    
        Args:
            error ( np.ndarray):  np.ndarray
            learning_rate (floar): float

        Returns:
            np.ndarray:  np.ndarray
        """

        return np.ones(self.X.size).reshape(self.X.shape)