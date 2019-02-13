# Module create_model
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Создадим контрольную точку при помощи callback функции
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 period=5)
model = create_model()

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # передаем callback обучению

# Сохраняем веса
model.save_weights('./checkpoints/my_checkpoint')

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
