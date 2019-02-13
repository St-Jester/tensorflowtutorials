from __future__ import absolute_import, division, print_function, unicode_literals

# Импортируем TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# А также добавим вспомогательные библиотеки для вычислений и вывода данных на экран

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# ============= Load datasets ==============

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============= Show initial images =========
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# convert to 0 to 1 range
train_images = train_images / 255.0

test_images = test_images / 255.0

# ============= Show prepared data ==========
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
#
# plt.show()

# Flatten - from 2d array yo 1d array [784]
# Dense - fully connected layer first has 128 neurons
# second - softmax - gives probability 0 to 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# ============== Compile model ===========

# Optimizer - how to update the mpdel according to input and loss
# Loss function = measures accuracy "guides the model"
# Metrics - how to measure accuracy

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# ================ Measure accuracy ==========
test_loss, test_acc = model.evaluate(test_images, test_labels)


# ============= Predict ============
predictions = model.predict(test_images)

# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)

img = test_images[0]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))

plt.show()
