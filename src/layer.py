import numpy as np


class Layer(object):
    def __init__(self, neurons_count, input_size):
        self.neurons_count = neurons_count
        self.input_size = input_size
        self.weights = np.random.rand(neurons_count, input_size)
        self.biases = np.random.rand(neurons_count, 1)
