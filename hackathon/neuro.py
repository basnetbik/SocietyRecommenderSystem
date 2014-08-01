__author__ = 'Pravesh'

import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

class Neuro:
    # Here are the parameters required by this class
    num_layers = 0
    layers = []     # layers should be all integers
    weights = []    # floating points
    n_i = 0         # number of input features
    n_o = 0         # number of output features
    train_input = None
    train_output = None
    regularization_param = 3.

    def __init__(self, layers):
        assert len(layers) != 0     # layers should be non-empty
        assert all([type(i) == int for i in layers])  # layers should be all integers
        self.layers = layers
        self.n_i = layers[0]
        self.n_o = layers[-1]
        self.num_layers = len(layers)

    def message(self, x):
        self.current_iteration += 1
        if self.current_iteration % 100 == 0:
            print "Iteration %d/%d" %(self.current_iteration, self.total_iteration)

    def train(self, input, output, epochs=50, regularization_param=1.0):
        """
        This routine is associated with training the neural network
        input : array-like inputs each of n-features where n_i = number of input-nodes
        output: array-like inputs each of m-features where n_o = number of output nodes
        """
        assert regularization_param >= 0
        self.regularization_param = regularization_param
        self.train_input = np.matrix(input)
        self.train_output = np.matrix(output)

        in_shape = self.train_input.shape
        out_shape = self.train_output.shape

        assert in_shape[0] != 0 and out_shape[0] != 0     # input and outputs are not empty
        assert in_shape[0] == out_shape[0]      # number of samples is consistent
        assert in_shape[1] == self.n_i     # number of features in input is consistent
        assert out_shape[1] == self.n_o  # number of features in output is consistent

        l = self.layers
        m = input.shape[0]  # number of training samples

        # initialize the weights
        np.random.seed()
        INIT_EPSILON = 1.
        self.weights = []
        for i in range(len(l) - 1):
            INIT_EPSILON = np.sqrt(6) / np.sqrt(l[i] + l[i + 1])
            t = 2 * INIT_EPSILON * np.random.random_sample(
                (l[i] + 1, l[i + 1])) - INIT_EPSILON     # generate random weights in [-1, 1]
            self.weights.append(np.matrix(t))

        wt = self.unpack(self.weights)

        # beta = 0.7 * self.layers[1] ** (1./self.layers[0])
        # norm = np.sqrt(np.dot(wt,wt))
        # wt *= beta/norm

        self.total_iteration = epochs
        self.current_iteration = 0
        self.weights = self.pack(fmin_cg(self.cost, wt, self.gradients,
                                         maxiter=epochs, disp=1, callback=self.message))


    def check_gradients(self, input, output):
        # first make copies of all of the weights (to fiddle with later)
        epsilon = 1E-4
        gradients = []
        new_weights = []
        for i in self.weights:
            new_weights.append(i)
            gradients.append(np.zeros(i.shape))

        #for each layer, for each row, for each column
        for l in range(len(new_weights)):
            for i in range(new_weights[l].shape[0]):
                for j in range(new_weights[l].shape[1]):
                    backup = new_weights[l][i, j]
                    new_weights[l][i, j] += epsilon
                    cost = self.cost(input, output, weights=new_weights)
                    new_weights[l][i, j] -= 2 * epsilon
                    cost -= self.cost(input, output, weights=new_weights)
                    cost /= (2 * epsilon)
                    gradients[l][i, j] = cost
                    new_weights[l][i, j] = backup

        return gradients



    def pack(self, weights):
        """
        Pack the 1D weights into list of matrices according to the nodes in different layers
        """
        packed_weights = []
        for l in range(self.num_layers - 1):  # not including the output layer
            req_data = (self.layers[l] + 1) * (self.layers[l + 1])
            # remove req_data number of entries from weights and resize it
            packed_weights.append(weights[:req_data].reshape((self.layers[l] + 1, self.layers[l + 1])))
            weights = weights[req_data:]    #trim
        assert len(weights) == 0
        return packed_weights

    def unpack(self, weights):
        unpacked_weights = np.array([])
        for w in weights:
            t = w.reshape((1, w.size))
            unpacked_weights = np.append(unpacked_weights, t)
        return unpacked_weights


    def gradients(self, weights):
        """
        The gradient function called by scipy.optimize.fmin_cg function.
        Gives the gradients at a specified weights. The gradients are computed using backpropagation
        """

        weights = self.pack(weights)
        gradients = []
        for i in range(self.num_layers - 1):
            # gradients.append(np.zeros((self.layers[i] + 1, self.layers[i + 1])))
            gradients.append(0)

        activations = self.inner_simulate(self.train_input, weights=weights)
        m = self.train_input.shape[0]

        # calculate deltas
        # for the last layer
        delta = activations[self.num_layers - 1] - self.train_output
        for l in range(self.num_layers - 2, -1, -1):  # excluding the input layers and the output layers
            a = activations[l]
            if type(gradients[l]) == int:
                gradients[l] = a.T * delta
            else:
                gradients[l] += a.T * delta

            # calculate delta for this layer
            delta = np.multiply(delta * weights[l].T, np.multiply(a, (1 - a)))

            # discard the first element
            delta = delta[0:, 1:]

        # finalize the gradients
        for i in range(len(gradients)):
            gradients[i][1:, :] += self.regularization_param * weights[i][1:, :]
            gradients[i] /= m

        return self.unpack(gradients)


    def cost(self, weights=None):
        """
        The cost function called by the scipy.optimize.fmin_cg function.
        ip: 1D array of weights
        ou: a scalar value that gives the estimated cost of the NN
        """
        if weights is None:
            weights = self.weights
        else:
            weights = self.pack(weights)

        h_theta = self.inner_simulate(self.train_input, weights=weights)[-1]
        m = self.train_input.shape[0]
        s = -np.sum(np.multiply(self.train_output,
                                np.log(h_theta)) + np.multiply((1 - self.train_output), (np.log(1 - h_theta))))
        # s = np.sum(np.abs(self.train_output-h_theta))
        # regularize
        s /= m
        for w in weights:
            s += self.regularization_param / 2. / m * np.sum(np.square(w[1:, :]))
        return s

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))    # the sigmoid function

    def inner_simulate(self, working, weights=None):
        """
        Simulates the neural network in forward direction without any checks. DO NOT CALL THIS FROM OUTSIDE
        input: array like input features
        """
        #working = input.copy()
        m = working.shape[0]      # number of training samples
        activations = []
        if not weights:
            weights = self.weights
        bias = np.ones((m,1))
        for i in range(self.num_layers - 1):
            working = np.column_stack([bias, working])
            activations.append(working)
            z = working * weights[i]
            working = self.sigmoid(z)

        activations.append(working)     # working set is the final product
        return activations      # return all the activations

    def simulate(self, input):
        """
        Use this module to calculate forward activations
        """

        # ensure that the neural network has been trained
        assert self.layers
        assert self.weights
        assert input.shape[1] == self.layers[0]

        return self.inner_simulate(input)[-1]

    def plot_validation_curve(self, input_train, output_train, input_validation, output_validation,
                            lambda_list = (0, 0.001, 0.003, 0.1, 0.3, 1, 3, 5, 7, 10), epochs=15):
        """
        This function will plot error values as a function of the regularization parameters
        """
        error_train = []
        error_val = []
        for i, l in enumerate(lambda_list):
            print
            print "Training the neural network with lambda = %f"%l
            self.train(input_train, output_train, epochs, l)

            #calculate errors for training sets
            self.train_input = input_train
            self.train_output = output_train
            self.regularization_param = 0.      # error is calculated by setting the regularization parameter 0
            error_train.append(self.cost()*5)

            #calculate errors for validation sets
            self.train_input = input_validation
            self.train_output = output_validation
            self.regularization_param = 0.
            error_val.append(self.cost()*5)

        train = plt.plot(lambda_list, error_train, label='Training Error')
        test = plt.plot(lambda_list, error_val, label='Validation Error')
        plt.xlabel("Lambda")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

    def plot_learning_curve(self, input_train, output_train, input_validation, output_validation,
                            lamda=1, epochs=15):
        """
        This function plots the learning curve which helps in determining the bias/variance of the model
        """
        error_train = []
        error_val = []
        m = input_train.shape[0]
        num_samples = []
        for i in range(1, m, 400):
            print
            print "Training the neural network with %d samples, %d remaining"%(i+1, m-i-1)
            self.train(input_train[0:i,:], output_train[0:i], epochs, regularization_param=lamda)
            num_samples.append(i)
            #calculate errors for training sets
            self.train_input = input_train[0:i,:]
            self.train_output = output_train[0:i]
            self.regularization_param = 0.      # error is calculated by setting the regularization parameter 0
            error_train.append(self.cost()*5)

            #calculate errors for validation sets
            self.train_input = input_validation
            self.train_output = output_validation
            self.regularization_param = 0.
            error_val.append(self.cost()*5)

        plt.plot(num_samples, error_train, label='Training Error')
        plt.plot(num_samples, error_val, label='Validation Error')
        plt.xlabel("Number of samples (m)")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

