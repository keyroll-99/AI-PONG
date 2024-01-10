import numpy as np

from ENV import WIDTH, HEIGHT, BALL_SPEED

DEFAULT_NUMBER_OF_EPOCH = 20_000


def sigmoid(x):
    x = np.clip(x, -450, 450)
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    x = np.clip(x, -450, 450)
    return x * (1 - x)


def prepare_data(train_data):
    train_data_result = []
    target_data_result = []

    for i in np.arange(0, len(train_data) - 1):
        train_data_result.append([
            train_data[i]["ball"]["x"] / WIDTH,
            train_data[i]["ball"]["y"] / HEIGHT,
            train_data[i]["ball"]["speed_x"] / BALL_SPEED,
            train_data[i]["ball"]["speed_y"] / BALL_SPEED,
            train_data[i]["player"]["y"] / HEIGHT,
            train_data[i]["opponent"]["y"] / HEIGHT,
        ])
        target_data_result.append([train_data[i]['player']['target']])

    return np.array(train_data_result), np.array(target_data_result)


class NeuralNetwork:
    def __init__(self, layers, learning_const):
        self.weights = []
        self.learning_const = learning_const
        self.layers = layers

        for i in np.arange(0, len(layers) - 1):
            weights = np.random.randn(layers[i] + 1, layers[i + 1] + 1)  # plus one because of bias
            self.weights.append(weights / np.sqrt(layers[i]))

    def train(self, train_data, number_of_epoch=DEFAULT_NUMBER_OF_EPOCH):
        train_data, target_data = prepare_data(train_data)

        train_data = np.c_[train_data, np.ones((train_data.shape[0]))]

        for epoch in np.arange(0, number_of_epoch):
            input_index = np.random.random_integers(0, len(train_data) - 1)
            train_input = train_data[input_index]
            target_input = target_data[input_index]
            self.train_part(train_input, target_input)

            if epoch == 0 or (epoch + 1) % 1000 == 0:
                # only for debug, it's should bo on other data sample
                loss = self.calculate_loss(train_input, target_data)
                print(f"INFO {epoch + 1}, loss {loss:7f}")

    def predict(self, inputs, should_add_bias=True):
        p = np.atleast_2d(inputs)

        if should_add_bias:
            p = np.c_[p, np.ones(p.shape[0])]

        r = [p]
        for layer in np.arange(0, len(self.weights)):
            result = sigmoid(np.dot(r[layer], self.weights[layer]))
            r.append(result)

        return r[-1]

    def calculate_loss(self, input_data, target_data):
        targets = np.atleast_2d(target_data)
        predictions = self.predict(input_data, False)[-1]
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss

    def train_part(self, train_input, target):
        layers_value = [np.atleast_2d(train_input)]

        # forward
        for layer in np.arange(len(self.weights)):
            layer_outputs = layers_value[layer].dot(self.weights[layer])
            layers_value.append(sigmoid(layer_outputs))

        # counting error
        error = layers_value[-1] - target
        layer_errors = [error * sigmoid_deriv(layers_value[-1])]

        for layer in np.arange(len(layers_value) - 2, 0, -1):
            delta = layer_errors[-1].dot(self.weights[layer].T)
            layer_errors.append(delta * sigmoid_deriv(layers_value[layer]))

        layer_errors = layer_errors[::-1]

        for layer in np.arange(0, len(self.weights)):
            self.weights[layer] += -self.learning_const * layers_value[layer].T.dot(layer_errors[layer])
