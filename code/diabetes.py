import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy

# задаем для воспроизводимости результатов

numpy.random.seed(2)

# загружаем датасет, соответствующий последним пяти годам до определение диагноза
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
# разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
X, Y = dataset[:, 0:8], dataset[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.5, random_state=5)

# создаем модели, добавляем слои один за другим
model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=8, activation='relu'))  # входной слой требует задать input_dim
model.add(keras.layers.Dense(15, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))  # сигмоида вместо relu для определения вероятности

# компилируем модель, используем градиентный спуск adam
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# обучаем нейронную сеть
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_test, y_test))

# оцениваем результат
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
