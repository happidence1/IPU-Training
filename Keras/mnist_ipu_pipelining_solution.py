# completed_demo_pipelining.py

# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright holder unknown (author: François Chollet 2015)
# Licensed under the Apache License, Version 2.0

import tensorflow.keras as keras
import numpy as np

from tensorflow.python import ipu

# STEP 1: Add multi-IPU and pipelining config variables
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 64
num_ipus = 2
num_replicas = num_ipus // 2
gradient_accumulation_steps_per_replica = 8


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


# STEP 2: Modify model_fn to use PipelineStage wrappers
def model_fn():
    input_layer = keras.Input(shape=input_shape)

    with keras.ipu.PipelineStage(0):
        x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)

    with keras.ipu.PipelineStage(1):
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x


def make_divisible(number, divisor):
    return number - number % divisor


# STEP 3: Adjust dataset sizes for pipelining requirements
(x_train, y_train), (x_test, y_test) = load_data()

train_data_len = x_train.shape[0]
train_steps_per_execution = train_data_len // (batch_size * num_replicas)
train_steps_per_execution = make_divisible(train_steps_per_execution, gradient_accumulation_steps_per_replica)
train_data_len = make_divisible(train_data_len, train_steps_per_execution * batch_size)
x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

test_data_len = x_test.shape[0]
test_steps_per_execution = test_data_len // (batch_size * num_replicas)
test_steps_per_execution = make_divisible(test_steps_per_execution, gradient_accumulation_steps_per_replica)
test_data_len = make_divisible(test_data_len, test_steps_per_execution * batch_size)
x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]

# STEP 4: Update IPU config to use multiple IPUs
ipu_config = ipu.config.IPUConfig()
ipu_config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
ipu_config.auto_select_ipus = num_ipus
ipu_config.configure_ipu_system()

# STEP 5: Use IPUStrategy
strategy = ipu.ipu_strategy.IPUStrategy()

print("Keras MNIST example, running on IPU with pipelining")
with strategy.scope():
    model = keras.Model(*model_fn())

    # STEP 6: Set pipelining options
    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica,
        pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped,
    )

    # STEP 7: Compile model for training with steps_per_execution
    model.compile(
        "sgd",
        "categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=train_steps_per_execution,
    )
    model.summary()

    print("\nTraining")
    model.fit(x_train, y_train, epochs=3, batch_size=batch_size)

    # STEP 8: Re-compile before evaluation with test steps_per_execution
    model.compile(
        "sgd",
        "categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=test_steps_per_execution,
    )

    print("\nEvaluation")
    model.evaluate(x_test, y_test, batch_size=batch_size)

print("Program ran successfully")

