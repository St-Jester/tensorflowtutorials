import create_model as cm

model = cm.create_model()

loss, acc = model.evaluate(cm.test_images, cm.test_labels)
print("Необученная модель, точность: {:5.2f}%".format(100 * acc))

# auto
# model.load_weights(cm.latest)

# hand
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(cm.test_images, cm.test_labels)
print("Восстановленная модель, точность: {:5.2f}%".format(100 * acc))
