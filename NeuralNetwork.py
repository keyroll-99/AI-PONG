import numpy as np
import random

from numpy import ndarray

INPUT_SIZE = 6
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    weights: np.ndarray
    output: list[float]
    bias: np.ndarray

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.array([[np.random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)])
        self.bias = np.array([np.random.uniform(-1, 1) for _ in range(output_size)])

    def predict(self, inputs):
        output = []
        i = 0
        for weights, bias in zip(self.weights, self.bias):
            output.append(sigmoid(sum(w * inp for w, inp in zip(weights, inputs)) + bias))

        return output

    def get_error(self, layer_value, prev_error: ndarray) -> ndarray:
        def count_error(weight: float):
            print("error", prev_error, weight, layer_value)
            for i in range(len(prev_error)):
                return prev_error[i] * weight * layer_value[i] * (1 - layer_value[i])

        errors = self.weights.copy()
        v_func = np.vectorize(count_error)
        v_func(errors)

        return errors


class NeuralNetwork:
    layers: list[Layer]
    iter: int

    def __init__(self, layers_count: int, hidden_layers_size: list[int]):
        layers_count += 1  # dodajemy warstwę wejściową
        self.layers = [Layer(INPUT_SIZE, hidden_layers_size[0])]
        for i in range(1, layers_count - 1):
            self.layers.append(Layer(hidden_layers_size[i - 1], hidden_layers_size[i]))
        self.layers.append(Layer(hidden_layers_size[-1], OUTPUT_SIZE))
        self.iter = 0

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.predict(inputs)
        return inputs[0]

    def train(self, inputs, target):
        # Forward pass
        layers_values = [inputs]
        for layer in self.layers:
            layers_values.append(layer.predict(layers_values[-1]))

        # Calculate errors
        output = layers_values[-1][0]

        errors = [[(output - target) * output * (1 - output)]]

        for i in range((len(self.layers) - 1), 0, -1):
            errors.append(self.layers[i].get_error(layers_values[i], errors[-1]))

        errors.reverse()
        # Update weights and biases
        for i in range(len(self.layers) - 1, 0, -1):
            if len(self.layers[i].weights[0]) != len(layers_values[i]):
                print("something went wrong", len(self.layers[i].weights[0]), len(layers_values[i]))
                raise Exception("something went wrong")
            layer_values = layers_values[i]
            for j, weights in enumerate(self.layers[i].weights):
                error = errors[i][j]
                for k, weight in enumerate(weights):
                    self.layers[i].weights[j][k] = weight + (LEARNING_RATE * error * layer_values[k])

            for j, bias in enumerate(self.layers[i].bias):
                self.layers[i].bias[j] = bias + (LEARNING_RATE * errors[i][j])

        print(f"{self.iter}-{len(self.layers)}")
        self.iter += 1
