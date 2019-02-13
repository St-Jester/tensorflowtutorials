import create_model as cm

model = cm.create_model()

# Используй keras.optimizer чтобы восстановить оптимизатор из файла HDF5
model.compile(optimizer=cm.keras.optimizers.Adam(),
              loss=cm.tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(cm.train_images, cm.train_labels, epochs=5)

# Сохраним модель полностью в единый HDF5 файл
model.save('my_model.h5')

# Воссоздадим точно такую же модель, включая веса и оптимизатор:
new_model = cm.keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(cm.test_images, cm.test_labels)
print("Восстановленная модель, точность: {:5.2f}%".format(100 * acc))
