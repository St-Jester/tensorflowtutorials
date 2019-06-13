from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
from tensorflow import set_random_seed
set_random_seed(2)
# задаем для воспроизводимости результатов
numpy.random.seed(2)

# загружаем датасет, соответствующий последним пяти годам до определение диагноза
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
# разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
X, Y = dataset[:, 0:8], dataset[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25, shuffle=False)

# создаем модели, добавляем слои один за другим
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(units=9, input_shape=(8, ), activation='relu',
                       name='input_layer'))  # входной слой требует задать input_dim
model.add(keras.layers.Dense(units=12, activation='relu', name='dense1'))
model.add(keras.layers.Dense(units=10, activation='relu', name='dense2'))
model.add(keras.layers.Dense(units=8, activation='relu', name='dense3'))
# model.add(keras.layers.Dense(units=8, activation='relu', name='dense4'))

model.add(keras.layers.Dense(units=1, activation='sigmoid',
                             name='output_layer'))  # сигмоида вместо relu для определения вероятности
# print(model.layers)
# print(model.inputs)
# print(model.outputs)
# # компилируем модель, используем градиентный спуск adam
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# обучаем нейронную сеть
history = model.fit(x_train, y_train, epochs=700, batch_size=10, validation_data=(x_test, y_test), verbose=0)

# оцениваем результат
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# filename = "../graphs/graph_{}.png".format(scores[1] * 100)
# plt.savefig(filename)
# plt.clf()
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
#
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# filename2 = "../graphs/graph_loss_{}.png".format(scores[1] * 100)
# plt.savefig(filename2)
