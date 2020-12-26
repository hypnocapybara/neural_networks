import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv('mushroom.csv')

    column_names = data.columns
    for names in column_names:
        data[names] = LabelEncoder().fit_transform(data[names].values)

    x_train = data.values
    y_train = to_categorical(data['edibility'])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(23,), activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val))

    print(model.evaluate(x_test, y_test))

    model_values = model.predict(x_test)

    for i, value_real in enumerate(y_test):
        value_model = model_values[i]
        if np.argmax(value_real) != np.argmax(value_model):
            print('NO EQUAL!')


if __name__ == '__main__':
    main()
