from re import M
from tkinter import N
import numpy as np
import sys
from si.statistics.sigmoid_function import sigmoid
sys.path.append("/Users/André Silva/SI/")
from si.data.dataset1 import Dataset
from si.metrics.mse import mse


class LogisticRegression:
    """
    The LogisticRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, stop_value:float = 0.0001):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.stop= stop_value

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history= {}
    
    def fit(self,dataset: Dataset, use_adaptive_alpha:bool= True):
        if use_adaptive_alpha:
            self._adaptive_fit(dataset)
        else:
            self._regular_fit(dataset)

    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: Logistic_reg
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero
            
            y_pred= sigmoid(y_pred)

            # computing and updating the gradient with the learning rate
            # vector shape (n_features)-> gradient[k] updates self.theta[k]
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            #Store the cost in a dictionary
            self.cost_history[i]=self.cost(dataset)
            #Stops the gradient if condition
            if i != 0 and (self.cost_history[i-1]-self.cost_history[i] < self.stop):
                break

        return self

    def _adaptive_fit(self,dataset: Dataset,use_adaptive_alpha: bool= True) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: Logistic_reg
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent

        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero
            
            y_pred= sigmoid(y_pred)

            # computing and updating the gradient with the learning rate
            # vector shape (n_features)-> gradient[k] updates self.theta[k]
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            #Store the cost in a dictionary
            self.cost_history[i]=self.cost(dataset)
            #Stops the gradient if condition
            if i != 0 and (self.cost_history[i-1]-self.cost_history[i] < self.stop):
                self.theta= self.theta/2

        return self


    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        y_pred= sigmoid(np.dot(dataset.X, self.theta) + self.theta_zero)
        mask= y_pred >= 0.5
        y_pred[mask]==1
        y_pred[~mask]==0
        return y_pred


    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        m,n= dataset.shape()
        y_pred = sigmoid(np.dot(dataset.X,self.theta)+self.theta_zero)
        cost= -((dataset.y*np.log(y_pred)+(1-dataset.y)*np.log(1)-y_pred))
        cost= np.sum(cost+self.l2_penalty/2*m)
        return cost
        

if __name__ == '__main__':
    # import dataset
    from si.data.dataset1 import Dataset
    from si.read_csv import read_csv
    #make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)
    dataset2= read_csv("cpu.csv",sep=",")

    #fit the model
    model = LogisticRegression()
    model._regular_fit(dataset_)
    
    #get coefs
    print(f"Parameters: {model.theta}")

    #compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    #compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]]),))
    print(f"Predictions: {y_pred_}")

    #cost dic
    for value in model.cost_history.keys():
        print(f"cost of {value+1} iteration is: {model.cost_history[value]}")

