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
        self.X= None
        

    def foward(self,input_data:np.ndarray)-> np.ndarray:
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
        error_to_propagate= np.dot(self.weights.T,error)
        #Update weights and bias
        self.weights-= learning_rate*np.dot(self.X.T,error)
        self.bias-= learning_rate * np.sum(error, axis=0)
        #Return error for previous layer
        error_to_propagate= np.dot(error_to_propagate,self.weights.T)
        return error_to_propagate

class SigmoidActivation:
    """_summary_
    Foward 
    """
    def __init__(self) -> None:
        self.X= None 

    def foward(self,input_data): 
        """_summary_

        Args:
            input_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.X= input_data
        return sigmoid(input_data)


    def backward(self,input_data,learning_rate):
        """_summary_

        Args:
            input_data (_type_): _description_
        """
        sigmoid_derivate= sigmoid(input_data)
        sigmoid_derivate=sigmoid_derivate*(1-sigmoid_derivate)
        return sigmoid_derivate

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

    def backprop(self,input_data:Dataset,error: float):
        """_summary_
        Backpropagation
        Args:
            input_data (Dataset): Dataset
            error (float): float
        """
        #We need to use the output of our layer, or foward to get the error 
        error_to_prop= input_data.X * (error-np.sum(error*input_data.X,axis=1,keepdims=True))
        return error_to_prop

    
class ReLUActivation:
    """_summary_
     Generats a foward based on ReLUActivation function
    """
    def __init__(self) -> None:
        pass

    def foward(self,input_data:Dataset):
        """Foward
        Args:
            input_data (Dataset): Dataset

        Returns:
            _type_: Returns an np.ndarray
        """
        
        return np.maximum(min(input_data.X),input_data.X)

    def backprop(self,input_data:Dataset,error:np.ndarray)-> np.ndarray:
        """backpropagation

        Args:
            input_data (Dataset): Dataset

        Returns: np.darray
        """

        for data in input_data.X:
            if data>0:
                input_data.X[data]= 1
            else:
                input_data.X[data]= 0
        
        error_to_prop= error * (input_data.X)
        return error_to_prop
