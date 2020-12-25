import math
from functools import reduce
from typing import List

from .neuron import Neuron


def mse_loss(real_val, compute_val):
    squares_list = [
        (real_val[i] - compute_val[i]) ** 2 / len(real_val)
        for i in range(len(real_val))
    ]
    return sum(squares_list)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


class Net:
    neurons: List[List[Neuron]] = []

    def __init__(self, structure: List[int]):
        assert len(structure) > 2

        for i in range(1, len(structure)):
            input_weights = [0.5] * structure[i-1]
            layer = [Neuron(input_weights, 0) for _g in range(structure[i])]
            self.neurons.append(layer)

    def feed_forward(self, inputs: List[float]) -> List[float]:
        assert len(self.neurons) > 0
        assert len(inputs) == len(self.neurons[0])

        next_inputs = inputs
        for layer in self.neurons:
            next_inputs = [layer[i].feed_forward(next_inputs) for i in range(len(layer))]

        return next_inputs

    def train(self, values: List[List[float]], answers: List[List[float]]):
        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for data, answer in zip(values, answers):
                neurons_summs = []
                neurons_results = []
                next_inputs = data
                for layer in self.neurons:
                    next_sums_layer = []
                    next_results_layer = []
                    for neuron in layer:
                        layer_sum = sum([
                            next_inputs[i] * neuron.weights[i]
                            for i in range(len(next_inputs))
                        ]) + neuron.bias
                        next_sums_layer.append(layer_sum)
                        next_results_layer.append(sigmoid(layer_sum))

                    neurons_summs.append(next_sums_layer)
                    neurons_results.append(next_results_layer)
                    next_inputs = next_results_layer

                # derivative for MSE of answers
                d_answer = -2 * math.prod([(answer[i] - next_inputs[i]) for i in range(len(answer))]) / len(answer)

                for i in reversed(range(len(self.neurons))):
                    layer = self.neurons[i]
                    for neuron_index, neuron in enumerate(layer):
                        inputs = neurons_results[i-1] if i > 0 else data
                        weights = neuron.weights
                        for g in range(len(weights)):
                            weights[g] -= learn_rate * d_answer * inputs[g] * deriv_sigmoid(neurons_summs[i][neuron_index])

                        neuron.bias -= learn_rate * d_answer * deriv_sigmoid(neurons_summs[i][neuron_index])
