import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.python.keras.optimizer_v2.adam import Adam


def main():
    df_train = pd.read_csv('wine.csv', sep=';')
    y_train = df_train['quality']

    # del df_train['quality']

    X_train = df_train.values
    X_train = normalize(X_train)

    y_train = to_categorical(y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(12, input_shape=(12,), activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dropout(0.3),
    #
    #     tf.keras.layers.Dense(6, activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dropout(0.3),
    #
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, input_shape=(12,), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['acc'])

    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=2)

    print(model.evaluate(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred_cl = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    diffs = [y_pred_cl[i] - y_true[i] for i in range(len(y_pred_cl))]
    print(diffs)

    # train_data = data[:len(data)-10]
    # check_data = data[len(data)-10:]
    #
    # values = []
    # answers = []
    #
    # for row in train_data:
    #     values.append([float(v) for v in row[1:10]])
    #     answer = [0.0] * 7
    #     index = int(row[10]) - 1
    #     answer[index] = 1.0
    #     answers.append(answer)
    #
    # model = Sequential([
    #     Dense(9, input_shape=(9,), activation='relu'),
    #     Dense(9, activation='relu'),
    #     Dense(7, activation='sigmoid'),
    # ])
        # model.compile(
        #     optimizer='adam',
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy'],
        # )
    # model.fit(values, answers, epochs=5)
    #
    # for check in check_data:
    #     row = [float(v) for v in check[1:10]]
    #     net_output = model.predict([row])
    #     net_output = [e for e in net_output[0]]
    #     print("net output", net_output, "real", check[10])

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
