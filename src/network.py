import numpy as np
from layer import Layer


class Network(object):
    def __init__(self, hidden_count, hidden_size, input_size, output_size, hidden_act_fun, output_act_fun):
        self.hidden_count = hidden_count
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_act_fun = hidden_act_fun
        self.output_act_fun = output_act_fun

        self.hidden_layers, self.output_layer = self.init_layers()

    def init_layers(self):
        hidden_layers = []
        for i in range(self.hidden_count):
            if i == 0:
                print('{}, {}'.format(self.hidden_size, self.input_size))
                layer = Layer(self.hidden_size, self.input_size)
            else:
                layer = Layer(self.hidden_size, self.hidden_size)
            hidden_layers.append(layer)

        output_layer = Layer(self.output_size, self.hidden_size)
        return hidden_layers, output_layer

    def forward_prop(self, batch):
        """
        batch: size = image_vector x batch_size (rows x cols)
        layer.weights: size = neurons x input_size
        hidden activation: size = neurons x batch_size
        output: size = output_size x batch_size
        """

        pre_activations = []
        activations = []

        curr_data = batch
        for layer in self.hidden_layers:
            data = layer.weights.dot(curr_data)
            data += layer.bias_vector
            pre_activations.append(data)
            curr_data = self.hidden_act_fun(data)
            activations.append(curr_data)

        data = self.output_layer.weights.dot(curr_data)
        data += self.output_layer.bias_vector
        output = self.output_act_fun(data)
        return pre_activations, activations, output

    def mean_batch_error(self, output, labels, batch_size):
        error = np.power(labels - output, 2)
        mean_error = error.sum(axis=1) / batch_size
        return mean_error

    def mean_batch(self, activations, pre_activations, batch_size):
        mean_activations = activations.sum(axis=1) / batch_size
        mean_pre_activations = pre_activations.sum(axis=1) / batch_size
        return mean_activations, mean_pre_activations
