from perceptron.neuron import Neuron
from perceptron.net import Net


def main():
    # weights = [0, 1]
    # bias = 0
    #
    # h1 = Neuron(weights, bias)
    # h2 = Neuron(weights, bias)
    # o1 = Neuron(weights, bias)
    #
    # out_h1 = h1.feed_forward([2, 3])
    # out_h2 = h2.feed_forward([2, 3])
    #
    # out = o1.feed_forward([out_h1, out_h2])
    # print(out)

    net = Net([2, 2, 1])
    print(net.feed_forward([2, 3]))


if __name__ == '__main__':
    main()
