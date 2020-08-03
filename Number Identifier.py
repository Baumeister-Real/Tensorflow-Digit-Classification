import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(image_train, answer_train), (image_test, answer_test) = tf.keras.datasets.mnist.load_data()

original_images = image_test

image_train, image_test = image_train / 255.0, image_test / 255.0
answer_train = tf.keras.utils.to_categorical(answer_train)
answer_test = tf.keras.utils.to_categorical(answer_test)
image_train = image_train.reshape(
    image_train.shape[0], image_train.shape[1], image_train.shape[2], 1
)
image_test = image_test.reshape(
    image_test.shape[0], image_test.shape[1], image_test.shape[2], 1
)

model = keras.Sequential([
    keras.layers.Conv2D(28,(3,3),activation = "relu", input_shape = (28,28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(28, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(image_train, answer_train, epochs=1)
prediction = model.predict(image_test)

for i in range(10):
   plt.grid(False)
   img = np.reshape(image_test[i], (28, 28))
   plt.imshow(img)
   plt.xlabel("Actual: " + str(np.where(answer_test[i])[0][0]))
   plt.title("Prediction: " + str(np.argmax(model.predict(image_test[i].reshape(-1, 28,28,1)))))
   plt.show()

#model.save("digit_model.h5")

