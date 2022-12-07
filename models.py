import tensorflow.keras.datasets as tfd
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow.keras.regularizers as tfr
import tensorflow as tf
import numpy as np

FC_SP_16_256 = tfm.Sequential([
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

FC_MP_16_256 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same"),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512 = tfm.Sequential([
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

VGG_2B_32_64 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu"),
    tfl.Dense(128, activation="relu"),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64 = tfm.Sequential([
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

VGG_3B_32_128 = tfm.Sequential([
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



FC_SP_16_256_batch_norm = tfm.Sequential([
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

FC_MP_16_256_batch_norm = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.MaxPool2D(),
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
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same"),
    tfl.BatchNormalization(),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_batch_norm = tfm.Sequential([
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

VGG_2B_32_64_dropout_05 = tfm.Sequential([
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
    tfl.Dense(256, activation="relu"),
    tfl.Dropout(0.5),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.5),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_dropout_05 = tfm.Sequential([
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

VGG_3B_32_128_dropout_05 = tfm.Sequential([
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


VGG_2B_32_64_dropout_04 = tfm.Sequential([
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
    tfl.Dropout(0.4),
    tfl.Dense(256, activation="relu"),
    tfl.Dropout(0.4),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.4),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_dropout_04 = tfm.Sequential([
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
    tfl.Dropout(0.4),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.4),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_dropout_04 = tfm.Sequential([
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
    tfl.Dropout(0.4),
    tfl.Dense(256, activation="relu"),
    tfl.Dropout(0.4),
    tfl.Dense(10, activation="softmax")
])


VGG_2B_32_64_dropout_03 = tfm.Sequential([
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
    tfl.Dropout(0.3),
    tfl.Dense(256, activation="relu"),
    tfl.Dropout(0.3),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.3),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_dropout_03 = tfm.Sequential([
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
    tfl.Dropout(0.3),
    tfl.Dense(128, activation="relu"),
    tfl.Dropout(0.3),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_dropout_03 = tfm.Sequential([
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
    tfl.Dropout(0.3),
    tfl.Dense(256, activation="relu"),
    tfl.Dropout(0.3),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l1_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l1_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l1_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l1_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l1_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l1_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l1_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l1_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l1_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l1_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.001)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1(0.001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l1_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1(0.001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l1_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.001)),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l1_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l1_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l1_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l1_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.01)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1(0.01)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l1_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1(0.01)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l1_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.01)),
    tfl.Dense(10, activation="softmax")
])




FC_SP_16_256_l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L2(0.0001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L2(0.001)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L2(0.001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L2(0.001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L2(0.01)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L2(0.01)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L2(0.01)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1(0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1(0.0001)),
    tfl.Dense(10, activation="softmax")
])



FC_SP_16_256_l1l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l1l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l1l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l1l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l1l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l1l2_00001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l1l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l1l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l1l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l1l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l1l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l1l2_0001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
    tfl.Dense(10, activation="softmax")
])


FC_SP_16_256_l1l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])                                                                               
])

FC_MP_16_256_l1l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

FC_MP_32_512_l1l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", padding="same"),
    tfl.Reshape([10])
])

VGG_2B_32_64_l1l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_16_64_l1l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(128, activation="relu", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Dense(10, activation="softmax")
])

VGG_3B_32_128_l1l2_001 = tfm.Sequential([
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.MaxPool2D(),
    tfl.Flatten(),
    tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
    tfl.Dense(10, activation="softmax")
])
