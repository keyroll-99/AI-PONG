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
    bias: float

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.array([[np.random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)])
        self.bias = np.random.uniform(-1, 1)

    def predict(self, inputs):
        # Jest problem z mam dwie różne wielkości tablicy i nie mogę ich pomnożyć (output_size_prev != input_size)
        return sigmoid(sum(w * inp for w, inp in zip(self.weights, inputs)) + self.bias)

    def get_error(self, layer_value, prev_error: ndarray) -> ndarray:
        def count_error(weight: float):
            for i in range(len(prev_error)):
                return sum(prev_error[i] * weight * layer_value[i] * (1 - layer_value[i]))

        errors = self.weights.copy()
        v_func = np.vectorize(count_error)
        v_func(errors)

        return errors


class NeuralNetwork:
    layers: list[Layer]
    iter: int

    def __init__(self, layers_count: int, hidden_layers_size: list[int]):
        self.layers = [Layer(INPUT_SIZE, hidden_layers_size[0])]
        for i in range(1, layers_count - 1):
            self.layers.append(Layer(hidden_layers_size[i - 1], hidden_layers_size[i]))
        self.layers.append(Layer(hidden_layers_size[-1], OUTPUT_SIZE))
        self.iter = 0;

    def predict(self, inputs):
        for layer in self.layers:
            inputs = [layer.predict(inputs)]
        return inputs[0]

    def train(self, inputs, target):
        # Forward pass
        layer_values = [inputs]
        for layer in self.layers:
            layer_values.append([layer.predict(layer_values[-1])])

        # Calculate errors
        errors = [target - layer_values[-1]]
        for i in range(len(self.layers) - 1, 0, -1):
            errors.append(self.layers[i].get_error(layer_values[i], errors[-1]))

        # Update weights and biases
        for i in range(len(self.layers)):
            self.layers[i].weights += LEARNING_RATE * errors[-i - 1] * layer_values[i]
            self.layers[i].bias += LEARNING_RATE * errors[-i - 1]

        print(f"{self.iter}-{len(self.layers)}")


# class NeuralNetwork:
#     def __init__(self):
#         # Initialize weights and biases
#         self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(INPUT_SIZE)] for _ in range(HIDDEN_SIZE)]
#         self.biases_hidden = [random.uniform(-1, 1) for _ in range(HIDDEN_SIZE)]
#         self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(HIDDEN_SIZE)]
#         self.bias_output = random.uniform(-1, 1)
#

# weights_input_hidden1 = [
#     [random.uniform(-1, 1) for _ in range(INPUT_SIZE)]
#     for _ in range(8)
# ]
#
#
# def predict(self, inputs):
#     # Forward pass
#     hidden1_layer = [
#         sigmoid(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
#         for weights, bias in zip(self.weights_input_hidden1, self.biases_hidden1)
#     ]
#
#     hidden2_layer = [
#         sigmoid(sum(w * h1 for w, h1 in zip(weights, hidden1_layer)) + bias)
#         for weights, bias in zip(self.weights_hidden1_hidden2, self.biases_hidden2)
#     ]
#
#     output = sigmoid(sum(w * h2 for w, h2 in zip(self.weights_hidden2_output, hidden2_layer)) + self.bias_output)
#     return output
#
#
# def train(self, inputs, target):
#     # Forward pass
#     hidden1_layer = [
#         sigmoid(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
#         for weights, bias in zip(self.weights_input_hidden1, self.biases_hidden1)
#     ]
#
#     hidden2_layer = [
#         sigmoid(sum(w * h1 for w, h1 in zip(weights, hidden1_layer)) + bias)
#         for weights, bias in zip(self.weights_hidden1_hidden2, self.biases_hidden2)
#     ]
#
#     output = sigmoid(sum(w * h2 for w, h2 in zip(self.weights_hidden2_output, hidden2_layer)) + self.bias_output)
#
#     # Calculate errors
#     output_error = target - output
#     hidden2_errors = [
#         output_error * self.weights_hidden2_output[i] * hidden2_layer[i] * (1 - hidden2_layer[i])
#         for i in range(HIDDEN_SIZE_2)
#     ]
#     hidden1_errors = []
#     for i in range(HIDDEN_SIZE_1):
#         for j in range(HIDDEN_SIZE_2):
#             hidden1_errors.append(
#                 sum(hidden2_errors[j] * self.weights_hidden1_hidden2[i][j] * hidden1_layer[i] * (1 - hidden1_layer[i])))
#
#     # Update weights and biases
#     self.weights_hidden2_output = [w + LEARNING_RATE * output_error * hidden2_layer[i] for i, w in
#                                    enumerate(self.weights_hidden2_output)]
#     self.bias_output += LEARNING_RATE * output_error
#
#     for i in range(HIDDEN_SIZE_2):
#         self.weights_hidden1_hidden2[i] = [
#             w + LEARNING_RATE * hidden2_errors[i] * hidden1_layer[j]
#             for j, w in enumerate(self.weights_hidden1_hidden2[i])
#         ]
#         self.biases_hidden2[i] += LEARNING_RATE * hidden2_errors[i]
#
#     for i in range(HIDDEN_SIZE_1):
#         self.weights_input_hidden1[i] = [
#             w + LEARNING_RATE * hidden1_errors[i] * inputs[j]
#             for j, w in enumerate(self.weights_input_hidden1[i])
#         ]
#         self.biases_hidden1[i] += LEARNING_RATE * hidden1_errors[i]
