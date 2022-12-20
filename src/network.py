import numpy as np
from layer import Layer
from random import shuffle


class Network(object):
    def __init__(
        self,
        hidden_count,
        hidden_size,
        input_size,
        output_size,
        hidden_act_fun,
        output_act_fun,
        learning_rate,
    ):
        self.hidden_count = hidden_count
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_act_fun = hidden_act_fun
        self.output_act_fun = output_act_fun
        self.learning_rate = learning_rate

        self.hidden_layers, self.output_layer = self.init_layers()

    def init_layers(self):
        hidden_layers = []
        for i in range(self.hidden_count):
            if i == 0:
                print("{}, {}".format(self.hidden_size, self.input_size))
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
        pre_output = data
        output = self.output_act_fun(data)
        return pre_activations, activations, pre_output, output

    def backward_prop(self, loss, pre_activations, activations, data):
        pass

    def update_params(self, layers_dWs, layers_dBs, output_dWs, output_dBs):
        """A method to update the parameters of the model.

        Args:
            layers_dWs (list): list of layers weights.
            layers_dBs (list): list of layers biases.
            output_dWs (np.array): array of output layer weights.
            output_dBs (np.array): array of output layer biases.
        """
        layers = self.hidden_layers
        for i in range(len(layers)):
            layers[i].weights -= layers_dWs[i] * self.learning_rate
            layers[i].bias -= layers_dBs[i] * self.learning_rate
        self.output_layer.weights -= output_dWs
        self.output_layer.bias -= output_dBs

    def fit(
        self, epochs, batch_size, train_data, train_labels, valid_data, valid_labels
    ):
        """A method to train the model.

        Args:
            epochs (int): number of epochs.
            batch_size (int): size of the batch.
            train_data (np.array): training data.
            train_labels (np.array): training labels.
            valid_data (np.array): validation data.
            valid_labels (np.array): validation labels.

        Returns:
            lists: lists of accuracies and losses for each epoch.
        """
        train_accs, train_losses = [], []
        valid_accs, valid_losses = [], []
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            shuffle(train_data)
            for i in range(batch_size, len(train_data), batch_size):
                data_sample = train_data[i - batch_size : i - 1]
                sample_labels = train_labels[i - batch_size : i - 1]
                pre_activations, activations, output = self.forward_prop(data_sample)
                layers_dWs, layers_dBs, output_dWs, output_dBs = self.backward_prop(
                    output - sample_labels,
                    pre_activations,
                    activations,
                )
                self.update_params(layers_dWs, layers_dBs, output_dWs, output_dBs)

            train_acc, train_loss = self.evaluate(train_data, train_labels, "train")
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            valid_acc, valid_loss = self.evaluate(valid_data, valid_labels, "valid")
            valid_accs.append(valid_acc)
            valid_losses.append(valid_loss)
        return train_accs, train_losses, valid_accs, valid_losses

    def predict(self, data):
        _, _, output = self.forward_prop(data)
        return output

    def get_predicted_labels(self, network_output):
        return np.argmax(network_output, axis=1)

    def get_accuracy(self, data, labels):
        predictions = self.get_predicted_labels(self.predict(data))
        return np.sum(predictions, axis=1) == labels / labels.size

    def get_loss(self, labels, network_output):
        batch_size = labels.shape[1]
        predictions = self.get_predicted_labels(network_output)
        error = np.power(labels - predictions, 2)
        mean_error = error.sum(axis=1) / batch_size
        return mean_error

    def evaluate(self, data, labels, data_part):
        accuracy = self.get_accuracy(data, labels)
        print("{} data accuracy: {:.2f}".format(data_part.capitalize(), accuracy))
        output = self.predict(data)
        loss = self.get_loss(labels, output)
        print("{} loss: {:.2f}".format(data_part, loss))
        return accuracy, loss

    def back_prop(self, input, pre_activations, activations, output_error):
        batch_size = output_error.size
        dz_out = output_error
        dw_out = dz_out.dot(activations[-1]) / batch_size
        db_out = np.sum(dz_out, 2) / batch_size

        dz_layers = []
        dw_layers = []
        db_layers = []

        for layer in range(self.hidden_count-1, -1, -1):
            if layer == self.hidden_count - 1:
                dz_layer = self.hidden_layers[layer].weights.dot(dz_out) * self.hidden_act_fun(pre_activations[layer], True)
            else:
                dz_layer = self.hidden_layers[layer].weights.dot(dz_layers[0]) * self.hidden_act_fun(pre_activations[layer], True)

            if layer != 0:
                dw_layer = dz_layer.dot(activations[layer-1]) / batch_size
            else:
                dw_layer = dz_layer.dot(input) / batch_size

            db_layer = np.sum(dz_layer, 2) / batch_size

            dz_layers.insert(0, dz_layer)
            dw_layers.insert(0, dw_layer)
            db_layers.insert(0, db_layer)

        return dw_out, db_out, dw_layers, db_layers
