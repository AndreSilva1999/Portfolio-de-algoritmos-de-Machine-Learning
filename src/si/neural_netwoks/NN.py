from si.data.dataset1 import Dataset
import numpy as np
import typing as Ml
from si.neural_netwoks.Layer import Dense,SigmoidActivation
from si.metrics.mse import mse_derivate,mse
from si.metrics.accuracy import accuracy
class NN:

    def __init__(self,layers: list,epochs: int = 1000,learning_rate:float=0.01,loss_function= mse,loss_derivate= mse_derivate,verbose= True) -> None:
        """_summary_
        Initializes parameters and atributes
        Args:
            layers (Ml): List
            epochs (int): Numero de iterações
            loss_function(Callable)= Loss function
            loss_derivate(Callable)= derivate of loss function

        """
        self.layers= layers
        self.epoc= epochs
        self.loss_function= loss_function
        self.loss_derivative= loss_derivate
        self.learning_rate= learning_rate
        self.verbose= verbose
        #Atributes
        self.fitted= False
        self.history={}


    def predict(self, dataset: Dataset) -> np.ndarray:
        """Predict for dataset

        Args:
            dataset (Dataset): _description_

        Returns:
            np.ndarray: ndarray
        """
        X = dataset.X.copy()

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def fit(self,dataset: Dataset):
        """It fits the model to the given dataset.
        Args:
            dataset (Dataset): Dataset
        """
        #Foward prop
        for i in range(1,self.epoc+1):
            y_true= np.array(dataset.X)
            y_pred= np.reshape(dataset.y,(-1,1))

            for layer in self.layers:
                y_true= layer.foward(y_true)

            #Backward prop

            error= self.loss_derivative(y_pred,y_true)
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)
            #Save cost
            cost= self.loss_function(y_pred,y_true)
            self.history[i]=cost

            if self.verbose:
                print(f"Epoch {i}/{self.epoc}-{cost=:.4f}")

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost of the model on the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on
        Returns
        -------
        cost: float
            The cost of the model
        """
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func = accuracy) -> float:
        """
        It computes the score of the model on the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on
        scoring_func: Callable
            The scoring function to use
        Returns
        -------
        score: float
            The score of the model
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)


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