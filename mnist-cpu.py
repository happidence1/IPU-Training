import tensorflow.keras as keras
import numpy as np

# Store class and shape information.
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 64


def load_data():
    # Load the MNIST dataset from keras.datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the images.
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # When dealing with images, we usually want an explicit channel dimension,
    # even when it is 1.
    # Each sample thus has a shape of (28, 28, 1).
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Finally, convert class assignments to a binary class matrix.
    # Each row can be seen as a rank-1 "one-hot" tensor.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def model_fn():
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)

    # Add layers to the graph.
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x


(x_train, y_train), (x_test, y_test) = load_data()

print('Keras MNIST example, running on CPU')
# Model.__init__ takes two required arguments, inputs and outputs.
model = keras.Model(*model_fn())

# Compile our model with Stochastic Gradient Descent as an optimizer
# and Categorical Cross Entropy as a loss.
model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])
model.summary()

print('\nTraining')
model.fit(x_train, y_train, epochs=3, batch_size=batch_size)

print('\nEvaluation')
model.evaluate(x_test, y_test, batch_size=batch_size)