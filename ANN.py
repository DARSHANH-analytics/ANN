import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dense, Flatten

mnist_data = keras.datasets.mnist
(X_train_full, y_train_full),(X_test, y_test) = mnist_data.load_data()

print(X_train_full.shape)
print(y_train_full.shape)
print(X_test.shape)
print(y_test.shape)

print('Y unique Values --> ',set(y_train_full))

print(X_train_full[0])
print('Label --> ',y_train_full[0])

print(X_train_full[0].max(), X_train_full[0].min())

# min values is 0 and max values is 255
# Lets further divide training data into train and validation sets
# divide the dataset by 255 to normalize bring all values between 0 and 1

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

#plt.imshow(X_train_full[0],cmap="binary")
#plt.show()

LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
]

model = tf.keras.models.Sequential(LAYERS)

print(model.layers)
print(model.summary())
print(model.layers[1].name)

weights, biases = model.layers[1].get_weights()

print(weights.shape,biases.shape)

LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ["accuracy"]

model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

EPOCHS = 30
VALIDATION = (X_valid, y_valid)

history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

print(pd.DataFrame(history.history))

pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]

y_prob = model.predict(X_new)

y_prob.round(3)

Y_pred= np.argmax(y_prob, axis=-1)
Y_pred

for img_array, pred, actual in zip(X_new, Y_pred, y_test[:3]):
  plt.imshow(img_array, cmap="binary")
  plt.title(f"predicted: {pred}, Actual: {actual}")
  plt.axis("off")
  plt.show()
  print("---"*20)

model.save("model.h5")