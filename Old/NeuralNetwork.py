import numpy as np
import random

from numpy import ndarray

from ENV import WIDTH, HEIGHT, BALL_SPEED

INPUT_SIZE = 6
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
EPOCH_TIME = 50_000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def get_input(data):
    return [
        data["ball"]["x"] / WIDTH,
        data["ball"]["y"] / HEIGHT,
        data["ball"]["speed_x"] / BALL_SPEED,
        data["ball"]["speed_y"] / BALL_SPEED,
        data["player"]["y"] / HEIGHT,
        data["opponent"]["y"] / HEIGHT,
    ]


class Layer:
    weights: np.ndarray
    output: list[float]
    bias: np.ndarray

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.array([[np.random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)])
        self.bias = np.array([np.random.uniform(-1, 1) for _ in range(output_size)])

    def to_json(self):
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist()
        }

    def predict(self, inputs):
        output = []
        for weights, bias in zip(self.weights, self.bias):
            output.append(sigmoid(np.dot(weights, inputs) + bias))

        return output

    def get_error(self, layer_values, prev_error: ndarray) -> ndarray:

        weights = self.weights.copy()
        errors = []

        for j, layer_value in enumerate(layer_values):
            error_sum = 0
            for k, error in enumerate(prev_error):
                error_sum += error * weights[k][j]

            errors.append(error_sum * sigmoid_derivative(layer_value))

        return np.array(errors)


class NeuralNetwork:
    layers: list[Layer]
    iter: int
    max_layer_size = 0

    def __init__(self, layers_count: int, hidden_layers_size: list[int], player_name: str):
        self.player_name = player_name
        layers_count += 1  # dodajemy warstwę wejściową
        self.layers = [Layer(INPUT_SIZE, hidden_layers_size[0])]
        for i in range(1, layers_count - 1):
            self.layers.append(Layer(hidden_layers_size[i - 1], hidden_layers_size[i]))
        self.layers.append(Layer(hidden_layers_size[-1], OUTPUT_SIZE))

        self.max_layer_size = max(hidden_layers_size)
        self.iter = 0

    def load_weights(self, weights):
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(weights[i]["weights"])
            layer.bias = np.array(weights[i]["bias"])

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.predict(inputs)
        return inputs[0]

    def train(self, data):
        current_number_of_iter = 0

        weight_delta = np.array(np.zeros((len(self.layers), self.max_layer_size, self.max_layer_size)))

        prev_error = 999999999
        while current_number_of_iter < EPOCH_TIME:
            test_case = random.choice(data)
            target = test_case["player"]["target"]
            test_case = get_input(test_case)

            layers_values = self.get_layers_value(test_case)
            output = layers_values[-1][0]
            errors = [[output - target * sigmoid_derivative(output)]]

            if errors[0][0] >= (1 + 0.01 * prev_error):
                weight_delta = np.array(np.zeros((len(self.layers), self.max_layer_size, self.max_layer_size)))
                continue

            errors = self.get_errors(errors, layers_values)

            # Update weights and biases
            prev_weights = [layer.weights.copy() for layer in self.layers]
            for i in range(len(self.layers) - 1, 0, -1):
                if len(self.layers[i].weights[0]) != len(layers_values[i]):
                    print("something went wrong", len(self.layers[i].weights[0]), len(layers_values[i]))
                    raise Exception("something went wrong")
                layer_values = layers_values[i]
                for j, weights in enumerate(self.layers[i].weights):
                    error = errors[i][j]
                    for k, weight in enumerate(weights):
                        self.layers[i].weights[j][k] = weight - (LEARNING_RATE * error * layer_values[k]) + \
                                                       weight_delta[i][j][k]
                        current_weight = self.layers[i].weights[j][k]
                        prev_weight = prev_weights[i][j][k]
                        weight_delta[i][j][k] = (current_weight - prev_weight) * 0.1

                for j, bias in enumerate(self.layers[i].bias):
                    self.layers[i].bias[j] = bias - (LEARNING_RATE * errors[i][j])

            current_number_of_iter += 1
            print(current_number_of_iter)

    def train_by_one_case(self, inputs, target):
        # Forward pass
        layers_values = self.get_layers_value(inputs)

        # Calculate errors
        output = layers_values[-1][0]

        error = output - target

        errors = [[error * sigmoid_derivative(output)]]

        errors = self.get_errors(errors, layers_values)
        # Update weights and biases
        for i in range(len(self.layers) - 1, 0, -1):
            if len(self.layers[i].weights[0]) != len(layers_values[i]):
                print("something went wrong", len(self.layers[i].weights[0]), len(layers_values[i]))
                raise Exception("something went wrong")
            layer_values = layers_values[i]
            for j, weights in enumerate(self.layers[i].weights):
                error = errors[i][j]
                for k, weight in enumerate(weights):
                    self.layers[i].weights[j][k] = weight - (LEARNING_RATE * error * layer_values[k])

            for j, bias in enumerate(self.layers[i].bias):
                self.layers[i].bias[j] = bias - (LEARNING_RATE * errors[i][j])
        self.iter += 1

    def get_layers_value(self, test_case):
        layers_values = [test_case]
        for layer in self.layers:
            layers_values.append(layer.predict(layers_values[-1]))
        return layers_values

    def get_errors(self, errors, layers_values):
        for i in range((len(self.layers) - 1), -1, -1):
            errors.append(self.layers[i].get_error(layers_values[i], errors[-1]))
        errors.reverse()
        return errors