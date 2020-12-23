from typing import List

from .neuron import Neuron


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
