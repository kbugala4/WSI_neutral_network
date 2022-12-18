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
                layer = Layer(self.hidden_size, self.input_size)
            else:
                layer = Layer(self.hidden_size, self.hidden_size)
            hidden_layers.append(layer)

        output_layer = Layer(self.output_size, self.hidden_size)
        return hidden_layers, output_layer

    def forward_prop(self, input):
        curr_data = input
        for layer in self.hidden_layers:
            data = np.matmul(layer.weights, curr_data)
            data += layer.bias_vector
            curr_data = self.hidden_act_func(data)

        data = np.matmul(self.output_layer.weights, curr_data)
        data += self.output_layer.bias_vector
        output = self.output_act_fun(data)
        return output
