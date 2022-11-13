import numpy as np
import random 

class NeuralNetwork:
    def __init__(self):
        self.input_layer_size = 10000
        self.output_layer_size = 1
    
    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))
    
    def setParameters(self, X, Y, hidden_layer_size):

        input_size = X.shape[0] # number of neuron in input layer
        output_size = Y.shape[0] # number of neuron in output layer.
        
        self.W1 = np.random.randn(self.input_layer_size, hidden_layer_size)
        self.b1 = np.zeros((hidden_layer_size, 1))
        self.W2 = np.random.randn(hidden_layer_size, self.output_layer_size)
        self.b2 = np.zeros((self.output_layer_size, 1))
        
        return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

    def forward_Propagation(self, X, parameters):
        
        self.Z1 = np.dot(X, parameters['W1']) + parameters['b1']
        self.A1 = self.sigmoid(Z1)
        
        self.Z2 = np.dot(self.A1, parameters['W2']) + parameters['b2']
        self.y = self.sigmoid(Z2)  
        return self.y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

    def cost(self, predict_val, actual_val):
        m = actual_val.shape[1]
        self.cost__ = -np.sum(np.multiply(np.log(predict_val), actual_val) + np.multiply((1 - actual_val), np.log(1 - predict_val)))/m
        return np.squeeze(cost__)
    
    def backPropagation(self, X, Y, parameters, cache):
        m = X.shape[1]
        self.dy = cache['y'] - Y
        
        self.dW2 = (1 / m) * np.dot(self.dy, np.transpose(cache['A1']))
        self.db2 = (1 / m) * np.sum(self.dy, axis=1)
        
        self.dZ1 = np.dot(np.transpose(parameters['W2']), self.dy) * (1-np.power(cache['A1'], 2))
        
        self.dW1 = (1 / m) * np.dot(self.dZ1, np.transpose(X))
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_Parameters(self, gradients, parameters, learning_rate = 0.05):

        self.W1 = parameters['W1'] - learning_rate * gradients['dW1']
        self.b1 = parameters['b1'] - learning_rate * gradients['db1']
        self.W2 = parameters['W2'] - learning_rate * gradients['dW2']
        self.b2 = parameters['b2'] - learning_rate * gradients['db2']
        return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

    def fit(self, X, Y, learning_rate, hidden_layer_size, number_of_iterations = 2000):
    
        self.parameters = self.setParameters(X, Y, hidden_layer_size)
    
        self.cost_ = []
    
        for j in range(number_of_iterations):
            self.y, self.cache = self.forward_Propagation(X, self.parameters)
        
            self.cost_it = self.cost(y, Y)
        
            self.gradients = self.backPropagation(X, Y, self.parameters, self.cache)
        
            self.parameters = self.update_Parameters(self.gradients, self.parameters, learning_rate)
        
            self.cost_.append(cost_it)
        return self.parameters, self.cost_
