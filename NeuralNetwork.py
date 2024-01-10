import numpy as np

from ENV import WIDTH, HEIGHT, BALL_SPEED

DEFAULT_NUMBER_OF_EPOCH = 20_000
HEIGHT_ALLOWED = 0.1


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
    def __init__(self, layers, learning_const, use_moment_method=False):
        self.weights = []
        self.learning_const = learning_const
        self.layers = layers
        self.use_moment_method = use_moment_method

        for i in np.arange(0, len(layers) - 1):
            weights = np.random.randn(layers[i] + 1, layers[i + 1] + 1)  # plus one because of bias
            self.weights.append(weights / np.sqrt(layers[i]))

    def to_json(self):
        r = []
        for layer in self.weights:
            r.append(layer.tolist())
        return r

    def from_json(self, json):
        self.weights = []
        for layer in json:
            self.weights.append(np.array(layer))

    def train(self, train_data, number_of_epoch=DEFAULT_NUMBER_OF_EPOCH):
        train_data, target_data = prepare_data(train_data)
        train_data = np.c_[train_data, np.ones((train_data.shape[0]))]
        moment_value = self.get_default_moment_method()
        prev_error = 0

        len_of_train_data = len(train_data)

        test_data = train_data[:int(len_of_train_data * 0.3)]
        test_target = target_data[:int(len_of_train_data * 0.3)]
        train_data = train_data[int(len_of_train_data * 0.3):]
        target_data = target_data[int(len_of_train_data * 0.3):]

        for epoch in np.arange(0, number_of_epoch):
            train_input, target_input = self.get_train_input(train_data, target_data)
            moment_value, prev_error = self.train_part(train_input, target_input, moment_value, prev_error)

            if epoch == 0 or (epoch + 1) % 1000 == 0:
                loss = self.calculate_loss(test_data, test_target)
                print(f"INFO {epoch + 1}, loss {loss:7f}")

    def get_train_input(self, train_data, target_data):
        input_index = np.random.random_integers(0, len(train_data) - 1)
        train_input = train_data[input_index]
        target_input = target_data[input_index]

        return train_input, target_input

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

    def train_part(self, train_input, target, moment_value, prev_error):
        layers_value = [np.atleast_2d(train_input)]

        # forward
        for layer in np.arange(len(self.weights)):
            layer_outputs = layers_value[layer].dot(self.weights[layer])
            layers_value.append(sigmoid(layer_outputs))

        # counting error
        error = layers_value[-1] - target
        flat_error = error[0][0]

        if flat_error < (1 + HEIGHT_ALLOWED) * prev_error and self.use_moment_method:
            return self.get_default_moment_method(), flat_error

        layer_errors = [error * sigmoid_deriv(layers_value[-1])]

        for layer in np.arange(len(layers_value) - 2, 0, -1):
            delta = layer_errors[-1].dot(self.weights[layer].T)
            layer_errors.append(delta * sigmoid_deriv(layers_value[layer]))

        layer_errors = layer_errors[::-1]
        new_weights = []
        for layer in np.arange(0, len(self.weights)):
            if self.use_moment_method:
                layer_shape = self.weights[layer].shape
                current_moment_value = []
                for i in np.arange(layer_shape[0]):
                    current_moment_value.append([])
                    for j in np.arange(layer_shape[1]):
                        current_moment_value[i].append(moment_value[layer][i][j])
                current_moment_value = np.array(current_moment_value)

                new_weights.append(self.weights[layer] + -self.learning_const * layers_value[layer].T.dot(
                    layer_errors[layer]) * self.learning_const + (current_moment_value * 0.1))

                moment_result = new_weights[layer] - self.weights[layer]
                for i, row in enumerate(moment_result):
                    for j, col in enumerate(row):
                        moment_value[layer][i][j] = col
            else:
                self.weights[layer] += -self.learning_const * layers_value[layer].T.dot(layer_errors[layer])

        if self.use_moment_method:
            self.weights = new_weights
        return moment_value, flat_error

    def get_default_moment_method(self):
        return np.zeros(shape=(len(self.layers), max(self.layers) + 1, max(self.layers) + 1))
