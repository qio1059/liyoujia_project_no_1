# 实现三层神经网络，包括前向传播、反向传播和参数更新。

import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu'):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size1) * 0.01,
            'b1': np.zeros((1, hidden_size1)),
            'W2': np.random.randn(hidden_size1, hidden_size2) * 0.01,
            'b2': np.zeros((1, hidden_size2)),
            'W3': np.random.randn(hidden_size2, output_size) * 0.01,
            'b3': np.zeros((1, output_size))
        }
        self.activation = activation

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.cache = {}
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = self.relu(self.cache['Z1']) if self.activation == 'relu' else np.tanh(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = self.relu(self.cache['Z2']) if self.activation == 'relu' else np.tanh(self.cache['Z2'])
        self.cache['Z3'] = np.dot(self.cache['A2'], self.params['W3']) + self.params['b3']
        self.cache['A3'] = self.softmax(self.cache['Z3'])
        return self.cache['A3']

    def backward(self, X, y, learning_rate=0.01, reg=0.01):
        m = X.shape[0]
        grads = {}

        y_onehot = np.zeros((m, 10))
        y_onehot[np.arange(m), y] = 1

        dZ3 = self.cache['A3'] - y_onehot
        grads['W3'] = np.dot(self.cache['A2'].T, dZ3) / m + reg * self.params['W3']
        grads['b3'] = np.sum(dZ3, axis=0, keepdims=True) / m
        dA2 = np.dot(dZ3, self.params['W3'].T)

        dZ2 = dA2 * (self.relu_derivative(self.cache['Z2']) if self.activation == 'relu' else 1 - np.tanh(self.cache['Z2'])**2)
        grads['W2'] = np.dot(self.cache['A1'].T, dZ2) / m + reg * self.params['W2']
        grads['b2'] = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.params['W2'].T)

        dZ1 = dA1 * (self.relu_derivative(self.cache['Z1']) if self.activation == 'relu' else 1 - np.tanh(self.cache['Z1'])**2)
        grads['W1'] = np.dot(X.T, dZ1) / m + reg * self.params['W1']
        grads['b1'] = np.sum(dZ1, axis=0, keepdims=True) / m

        for param in self.params:
            self.params[param] -= learning_rate * grads[param]
