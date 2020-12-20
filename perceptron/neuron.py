import math
from typing import List


class Neuron:
    def __init__(self, weights: List[float], bias: float):
        self.weights = weights
        self.bias = bias

    def activation_function(self, val):
        # sigmoid
        return 1 / (1 + math.exp(-val))

    def feed_forward(self, inputs: List[float]):
        assert len(inputs) == len(self.weights)
        return self.activation_function(
            sum([inputs[i] * self.weights[i] for i in range(len(inputs))])
        )
