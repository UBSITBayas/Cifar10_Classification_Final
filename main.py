import tensorflow as tf
import matplotlib.pyplot as plt



from tensorflow.keras import Sequential, datasets
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

y_train = y_train.reshape(-1, )
y_classes = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(len(y_classes))
def showImage(x, y, index):
     plt.figure(figsize=(15, 2))
     plt.imshow(x[index])
     plt.xlabel(y_classes[y[index]])
showImage(x_train, y_train, 7)
plt.show()

#normalize

x_train = x_train/255
x_test = x_test/255
# print(x_train[0])

# Build Model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), activation = "relu", input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (4, 4), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units = 34, activation = "relu"))
model.add(Dense(units = 10, activation = "softmax"))

#compile

model.compile(
     optimizer = "adam",
     loss = "sparse_categorical_crossentropy",
     metrics = ["accuracy"]
)

#train model

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=8)

loss, accuracy = model.evaluate(x_test, y_test)


# save model as tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("cifar10model.tflite", 'wb') as f:
     f.write(tflite_model)

'''
model.save('image_clasifier.tf')
model = keras.models.load_model('image_clasiificatier.tf')

# predictions
y_predictions = model.predict(x_test)
print(y_predictions[9])

y_predictions = [np.argmax(arr) for arr in y_predictions]
print(y_predictions)

y_test = y_test.reshape(-1, )
y_predictions[4]
showImage(x_test, y_test, 3)
plt.show()

# evaluate
model.evaluate(x_test, y_test)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", 'wb') as f:
     f.write(tflite_model)
'''
