import numpy as np


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return (x > 0).astype(float)


def cross_entropy_loss(y, y_hat):
    epsilon = 1e-9
    return -np.sum(y * np.log(y_hat + epsilon))


# cross entropy loss + softmax activation
def gradient_layer_3(batch_size, y, y_hat, A2, W3):
    assert y.shape == (10, batch_size)
    assert y_hat.shape == (10, batch_size)
    assert A2.shape == (64, batch_size)
    assert W3.shape == (10, 64)

    dC_dZ3 = (y_hat - y)
    assert dC_dZ3.shape == (10, batch_size)

    weight_gradient = (dC_dZ3 @ A2.T) / batch_size
    assert weight_gradient.shape == (10, 64)

    bias_gradient = np.sum(dC_dZ3, axis=1, keepdims=True) / batch_size
    assert bias_gradient.shape == (10, 1)

    propagator = W3.T @ dC_dZ3
    assert propagator.shape == (64, batch_size)

    return weight_gradient, bias_gradient, propagator


# ReLU activation
def gradient_layer_2(batch_size, dC_dA2, A2, A1, W2):
    assert dC_dA2.shape == (64, batch_size)
    assert A2.shape == (64, batch_size)
    assert A1.shape == (128, batch_size)
    assert W2.shape == (64, 128)

    dA2_dZ2 = derivative_relu(A2)
    dC_dZ2 = dC_dA2 * dA2_dZ2

    weight_gradient = (dC_dZ2 @ A1.T) / batch_size
    assert weight_gradient.shape == (64, 128)

    bias_gradient = np.sum(dC_dZ2, axis=1, keepdims=True) / batch_size
    assert bias_gradient.shape == (64, 1)

    propagator = W2.T @ dC_dZ2
    assert propagator.shape == (128, batch_size)

    return weight_gradient, bias_gradient, propagator
    

# ReLU activation
def gradient_layer_1(batch_size, dC_dA1, A1, A0, W1):
    assert dC_dA1.shape == (128, batch_size)
    assert A1.shape == (128, batch_size)
    assert A0.shape == (784, batch_size)
    assert W1.shape == (128, 784)

    dA1_dZ1 = derivative_relu(A1)
    dC_dZ1 = dC_dA1 * dA1_dZ1
    assert dC_dZ1.shape == (128, batch_size)

    weight_gradient = (dC_dZ1 @ A0.T) / batch_size
    assert weight_gradient.shape == (128, 784)

    bias_gradient = np.sum(dC_dZ1, axis=1, keepdims=True) / batch_size
    assert bias_gradient.shape == (128, 1)

    return weight_gradient, bias_gradient


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


# 784 input nodes
# 2 hidden layers with 128, 64 nodes
# 10 output nodes
class NeuralNetwork:
    def __init__(self):
        self.node_counts = [784, 128, 64, 10]
        self.weights = {}
        self.biases = {}

        # initialize weights
        for i in range(1, len(self.node_counts)):
            self.weights[i] = np.random.randn(self.node_counts[i], self.node_counts[i - 1]) * np.sqrt(2 / self.node_counts[i - 1])

        # initialize biases
        for i in range(1, len(self.node_counts)):
            self.biases[i] = np.random.rand(self.node_counts[i], 1)
        
        self.cache = {}

    def forward(self, inp):
        # input is 784 x m
        # weights[1] is 128 x 784
        # biases[1] is 128 x 1
        # Feed it through each layer with ReLU at hidden layers and softmax at output layer
        assert self.weights[1].shape == (128, 784)
        assert self.biases[1].shape == (128, 1)
        assert self.weights[2].shape == (64, 128)
        assert self.biases[2].shape == (64, 1)
        assert self.weights[3].shape == (10, 64)
        assert self.biases[3].shape == (10, 1)

        batch_size = inp.shape[1]

        self.cache = {}

        self.cache[0] = inp

        A1 = relu(self.weights[1] @ inp + self.biases[1])
        assert A1.shape == (128, batch_size)
        self.cache[1] = A1

        A2 = relu(self.weights[2] @ A1 + self.biases[2])
        assert A2.shape == (64, batch_size)
        self.cache[2] = A2

        A3 = softmax(self.weights[3] @ A2 + self.biases[3])
        assert A3.shape == (10, batch_size)
        self.cache[3] = A3

        return A3
    
    def backward(self, y, step):
        assert 0 in self.cache
        assert 1 in self.cache
        assert 2 in self.cache
        assert 3 in self.cache

        A0 = self.cache[0]
        A1 = self.cache[1]
        A2 = self.cache[2]
        A3 = self.cache[3]

        W1 = self.weights[1]
        W2 = self.weights[2]
        W3 = self.weights[3]
        y_hat = A3

        batch_size = A0.shape[1]

        # Layer 3
        weight_gradient, bias_gradient, propagator = gradient_layer_3(batch_size, y, y_hat, A2, W3)
        self.weights[3] -= weight_gradient * step
        self.biases[3] -= bias_gradient * step

        # Layer 2
        weight_gradient, bias_gradient, propagator = gradient_layer_2(batch_size, propagator, A2, A1, W2)
        self.weights[2] -= weight_gradient * step
        self.biases[2] -= bias_gradient * step

        # Layer 1
        weight_gradient, bias_gradient = gradient_layer_1(batch_size, propagator, A1, A0, W1)
        self.weights[1] -= weight_gradient * step
        self.biases[1] -= bias_gradient * step

