import tensorflow.keras.datasets as tfd
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow as tf
import numpy as np

full_conv = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

model_full_conv_max_pool = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same"),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_inspired_less_kernels = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu"),
    tfl.Dense(10, activation="softmax")
])

VGG_inspired_more_kernels = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu"),
    tfl.Dense(10, activation="softmax")
])

VGG_inspired_3_dense_layers = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Flatten(),
    tfl.Dense(512, activation="relu"),
    tfl.Dense(128, activation="relu"),
    tfl.Dense(10, activation="softmax")
])



full_conv_batch_norm = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

model_full_conv_max_pool_batch_norm = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_inspired_less_kernels_droput_05 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dropout(0.5),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.5),
    tfl.Dense(10, activation="softmax")
])

VGG_inspired_more_kernels_droput_05 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dropout(0.5),
    tfl.Dense(256, activation="relu"),
    tfl.Dropout(0.5),
    tfl.Dense(10, activation="softmax")
])

VGG_inspired_3_dense_layers_droput_05 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Flatten(),
    tfl.Dropout(0.5),
    tfl.Dense(512, activation="relu"),
    tfl.Dropout(0.5),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.5),
    tfl.Dense(10, activation="softmax")
])
