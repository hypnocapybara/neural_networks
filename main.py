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
    train_data = [
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ]
    answers = [[1.0], [0.0], [0.0], [1.0]]
    net.train(train_data, answers)

    print(net.feed_forward([-7, -3]))
    print(net.feed_forward([20, 2]))


if __name__ == '__main__':
    main()
