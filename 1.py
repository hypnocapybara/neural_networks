import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


SIZE = 224


def resize_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img, label


def train_and_save(model: tf.keras.Sequential):
    train, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
    train_resized = train[0].map(resize_image)
    train_batches = train_resized.shuffle(1000).batch(16)

    model.fit(train_batches, epochs=2)
    model.save('./model.h5')


def get_model():
    base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
    base_layers.trainable = False
    model = tf.keras.Sequential([
        base_layers,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


def main():
    if os.path.exists('./model.h5'):
        model = tf.keras.models.load_model('./model.h5')
    else:
        model = get_model()
        train_and_save(model)

    files = [f for f in os.listdir('images')]
    for f in files:
        img = load_img(os.path.join('images', f))
        img_array = img_to_array(img)
        img_resized, _ = resize_image(img_array, '')
        img_expended = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_expended)[0][0]
        prediction_type = 'cat' if prediction < 0.5 else 'dog'
        print(f, prediction_type, f'({prediction})')


if __name__ == '__main__':
    main()
