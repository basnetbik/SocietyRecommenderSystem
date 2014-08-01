"""
A simple linear regression module that implements a linear regression model as per specified by Prof Andrew Ng in his
coursera lectures.
"""

import numpy as np
from features_change import features_change
from scipy.optimize import fmin_cg



class linreg:

    #These values are used by the cost and gradient functions (since we don't pass them as parameters)
    X = None
    y = None

    #After training the linear regression model, we'll store the value in this vector
    theta = None
    regularization_param = 0.
    normalized = False
    xmu, xsigma = 0, 1
    ymu, ysigma = 0, 1

    def __init__(self):
        pass

    def train(self, X, y, regularization_param=1.0, alpha=0.001, epochs = 1500, normalize = True):
        """
        X: It is the input feature vector of shape m x n
        y: The output vector of shape m x 1
        regularization_param is the lambda, and alpha is the learning rate

        we'll try different algorithms here.. perhaps the most basic one will be gradient descent and that is what
        I'll be doing first.
        """

        if not X.ndim == 2:
            raise AssertionError("X must be a 2D vector")

        #Both X and y should be converted to matrix, that way it'll be easy to calculate cost and gradient
        if not isinstance(X, np.matrix):
            X = np.matrix(X)
        if not isinstance(y, np.matrix):
            y = np.matrix(y)

        #if y was given as a 1D row vector, then we convert it into a 1D column vector by taking its transpose to make
        #it consistent with our formulation of cost function and the gradient
        if y.ndim == 1:
            y = y.T

        # some sanity checks first
        m, n = X.shape
        assert m!=0 and n!=0, "Both training samples and number of features must be non-zero"
        assert y.shape[0] == m, "The training sample number must be same in both X and y"

        if normalize:
            X, self.xmu, self.xsigma = features_change(X, full_output=True)
            y, self.ymu, self.ysigma = features_change(y, full_output=True)
            self.normalized = True

        #now save X and y so that they may be visible from both the cost and the gradient function
        self.X = np.column_stack((np.ones_like(y), X))    #add the bias
        self.y = y

        #initialize all theta to 0 (another approach can be to initialize theta to random), including the bias
        self.theta = np.matrix(np.zeros((n+1,1)))
        self.regularization_param = self.regularization_param

        self.theta = self.gradient_descent(alpha=alpha, epochs=epochs)
        # print self.cost()

    def gradient_descent(self, alpha, epochs):
        """
        The gradient descent algorithm. Returns the parameter theta for the model
        """
        print self.cost()
        # for i in xrange(epochs):
            # print self.cost()
            # self.theta = self.theta - alpha*self.gradient(self.theta)
        self.theta = fmin_cg(self.cost, np.array(self.theta).reshape(self.theta.size), self.gradient, maxiter=epochs)
        self.theta = np.matrix(self.theta).T
        print self.cost()
        print
        return self.theta

    def cost(self, theta = None, X = None, y = None):
        """
        The cost function will calculate the least square error of the model
        """
        if theta is None:
            theta = self.theta
        else:
            theta = np.matrix(theta).T
        if X is None or y is None:
            X, y = self.X, self.y

        assert theta is not None and X is not None and y is not None, \
            "You must first call the train function before calling the cost function"

        m, n = X.shape
        m = 1
        ht = X * theta
        return 1./2./m * (np.sum( np.square(ht - y) ) + self.regularization_param* theta.T*theta)
        pass

    def gradient(self, theta):
        """
        gradient function will be called from the optimization algorithm.
        """
        assert theta is not None, "Theta is not initialized"
        theta = np.matrix(theta).T
        ht = self.X * theta - self.y
        grad = self.X.T * ht
        #also regularize the parameters except the bias
        grad[1:] += self.regularization_param*self.theta[1:]
        # return np.array(grad/self.X.shape[0]).reshape(theta.size)
        return np.array(grad).reshape(theta.size)

    def predict(self, X):
        X = np.matrix(X)
        m, n = 0, 0
        if X.ndim == 1:
            n = X.shape[0]
            m = 1
        else:
            m, n = X.shape
        assert m!=0 and n!=0, "Input must be non-empty"

        if self.normalized:
            X = (X-self.xmu)/self.xsigma

        X = np.column_stack((np.ones((m, 1)), X))
        assert self.X.shape[1] == X.shape[1], "The input features must match the output features"
        y = X * self.theta
        if self.normalized:
            y = y*self.ysigma + self.ymu
        return y