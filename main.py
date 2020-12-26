import random
from csv import reader

from perceptron.net import Net


def main():
    data = [row for row in reader(open('sonar.csv', 'r'))]
    random.shuffle(data)

    train_data = data[:len(data)-10]
    check_data = data[len(data)-10:]

    values = []
    answers = []

    for row in train_data:
        values.append([float(v) for v in row[:60]])
        answer = 1.0 if row[60] == 'R' else 0.0
        answers.append([answer])

    net = Net([60, 60, 10, 1])
    net.train(values, answers)

    for row in check_data:
        value = [float(v) for v in row[:60]]
        real_answer = 1.0 if row[60] == 'R' else 0.0
        net_answer = net.feed_forward(value)
        print('Real answer: %s | Net answer: %s' % (real_answer, net_answer))

    # net = Net([2, 2, 1])
    # train_data = [
    #     [-2, -1],  # Alice
    #     [25, 6],  # Bob
    #     [17, 4],  # Charlie
    #     [-15, -6],  # Diana
    # ]
    # answers = [[1.0], [0.0], [0.0], [1.0]]
    # net.train(train_data, answers)
    #
    # print(net.feed_forward([-7, -3]))
    # print(net.feed_forward([20, 2]))


if __name__ == '__main__':
    main()
