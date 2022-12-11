from si.data.dataset1 import Dataset
import numpy as np
import typing as Ml
from si.neural_netwoks.Layer import Dense,SigmoidActivation
from si.metrics.mse import mse
class NN:

    def __init__(self,layers: Ml(tuple,list),epochs: int = 1000,loss_function= mse,loss_derivate= mse_derivate,verbose= True) -> None:
        """_summary_
        Initializes parameters and atributes 
        Args:
            layers (Ml): _description_
            epochs (int): Numero de iterações
            loss_function(Callable)= Loss function
            loss_derivate(Callable)= derivate of loss function 

        """
        self.layers= layers
        self.epoc= epochs
        self.loss_function= loss_function
        self.loss_derivative= mse_derivate
        self.verbose= verbose
        #Atributes
        self.fitted= False
        self.history={}

    def fit(self,dataset:Dataset)-> np.ndarray:
        """_summary_
        Does foward for each layer

        Args:
            dataset (Dataset): _description_

        Returns:
            np.ndarray: _description_
        """
        x= dataset.X.copy()#Apontador oara o dataset.x e copia este valor 
        for layer in self.layers:
            x= layer.foward(x)
        self.fitted= True
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:

        X = dataset.X.copy()

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def fit(self):

        for i in range(1,self.epoc+1):

            for layer in self.layers:
                X= layers.foward(X)

            #Backward prop

            error= self.loss_derivative(y,X)
            for layer in self.layers[::-1]:
                error= layer.backward(error,self.learning_rate)

            cost= self.loss_function(y,X)
            self.history[i]=cost

            if self.verbose:
                print(f"Epoch {i}/{self.epoc}-{cost=:.4f}")

            


if __name__== "__main__":
    l1= Dense(input_size=2,output_size=2)
    l2= Dense(input_size=2,output_size=1)
    layers =[l1,l2]
    x= np.array([[2,1],[1,2],[1,1]])
    y= np.array([1,0,1])
    dataset= Dataset(X=x,y=y)
    l1_sg=SigmoidActivation()
    l2_sg= SigmoidActivation()

    nn_model= NN(layers=[l1,l1_sg,l2,l2_sg])

    nn_model.fit(dataset=dataset)
    nn_model.predict()