import tensorflow.keras.datasets as tfd
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow.keras.regularizers as tfr
import sklearn.metrics as skm
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

class Models():
    def __init__(self):
        self.FC_SP_16_256 = tfm.Sequential([
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

        self.FC_MP_16_256 = tfm.Sequential([
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

        self.FC_MP_32_512 = tfm.Sequential([
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

        self.VGG_2B_32_64 = tfm.Sequential([
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

        self.VGG_3B_16_64 = tfm.Sequential([
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

        self.VGG_3B_32_128 = tfm.Sequential([
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



        self.FC_SP_16_256_batch_norm = tfm.Sequential([
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

        self.FC_MP_16_256_batch_norm = tfm.Sequential([
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

        self.FC_MP_32_512_batch_norm = tfm.Sequential([
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

        self.VGG_2B_32_64_dropout_05 = tfm.Sequential([
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

        self.VGG_3B_16_64_dropout_05 = tfm.Sequential([
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

        self.VGG_3B_32_128_dropout_05 = tfm.Sequential([
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


        self.VGG_2B_32_64_dropout_04 = tfm.Sequential([
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

        self.VGG_3B_16_64_dropout_04 = tfm.Sequential([
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

        self.VGG_3B_32_128_dropout_04 = tfm.Sequential([
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


        self.VGG_2B_32_64_dropout_03 = tfm.Sequential([
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

        self.VGG_3B_16_64_dropout_03 = tfm.Sequential([
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

        self.VGG_3B_32_128_dropout_03 = tfm.Sequential([
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


        self.FC_SP_16_256_l1_00001 = tfm.Sequential([
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

        self.FC_MP_16_256_l1_00001 = tfm.Sequential([
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

        self.FC_MP_32_512_l1_00001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l1_00001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l1_00001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l1_00001 = tfm.Sequential([
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


        self.FC_SP_16_256_l1_0001 = tfm.Sequential([
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

        self.FC_MP_16_256_l1_0001 = tfm.Sequential([
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

        self.FC_MP_32_512_l1_0001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l1_0001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l1_0001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l1_0001 = tfm.Sequential([
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


        self.FC_SP_16_256_l1_001 = tfm.Sequential([
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

        self.FC_MP_16_256_l1_001 = tfm.Sequential([
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

        self.FC_MP_32_512_l1_001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l1_001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l1_001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l1_001 = tfm.Sequential([
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




        self.FC_SP_16_256_l2_00001 = tfm.Sequential([
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

        self.FC_MP_16_256_l2_00001 = tfm.Sequential([
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

        self.FC_MP_32_512_l2_00001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l2_00001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l2_00001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l2_00001 = tfm.Sequential([
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


        self.FC_SP_16_256_l2_0001 = tfm.Sequential([
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

        self.FC_MP_16_256_l2_0001 = tfm.Sequential([
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

        self.FC_MP_32_512_l2_0001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l2_0001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l2_0001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l2_0001 = tfm.Sequential([
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


        self.FC_SP_16_256_l2_001 = tfm.Sequential([
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

        self.FC_MP_16_256_l2_001 = tfm.Sequential([
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

        self.FC_MP_32_512_l2_001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l2_001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l2_001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l2_001 = tfm.Sequential([
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



        self.FC_SP_16_256_l1l2_00001 = tfm.Sequential([
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

        self.FC_MP_16_256_l1l2_00001 = tfm.Sequential([
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

        self.FC_MP_32_512_l1l2_00001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l1l2_00001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l1l2_00001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l1l2_00001 = tfm.Sequential([
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


        self.FC_SP_16_256_l1l2_0001 = tfm.Sequential([
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

        self.FC_MP_16_256_l1l2_0001 = tfm.Sequential([
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

        self.FC_MP_32_512_l1l2_0001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l1l2_0001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l1l2_0001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l1l2_0001 = tfm.Sequential([
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


        self.FC_SP_16_256_l1l2_001 = tfm.Sequential([
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

        self.FC_MP_16_256_l1l2_001 = tfm.Sequential([
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

        self.FC_MP_32_512_l1l2_001 = tfm.Sequential([
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

        self.VGG_2B_32_64_l1l2_001 = tfm.Sequential([
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

        self.VGG_3B_16_64_l1l2_001 = tfm.Sequential([
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

        self.VGG_3B_32_128_l1l2_001 = tfm.Sequential([
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


    def train(self, models_to_train, x_train, y_train):
        # ====================================================== baseline ======================================================
        if "baseline" in models_to_train or "all" in models_to_train:
            print("\n\nbaseline models training:")
            self.FC_SP_16_256.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256.save_weights("./models/FC_SP_16_256/FC_SP_16_256")
            del self.FC_SP_16_256

            self.FC_MP_16_256.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256.save_weights("./models/FC_MP_16_256/FC_MP_16_256")
            del self.FC_MP_16_256

            self.FC_MP_32_512.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512.save_weights("./models/FC_MP_32_512/FC_MP_32_512")
            del self.FC_MP_32_512

            self.VGG_2B_32_64.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64")
            del self.VGG_2B_32_64

            self.VGG_3B_16_64.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64")
            del self.VGG_3B_16_64

            self.VGG_3B_32_128.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128")
            del self.VGG_3B_32_128

        # ================================================= batch norm, dropout 0.3 =================================================
        if "dropout_03" in models_to_train or "all" in models_to_train:
            print("\n\nbatch norm, dropout_03 models training:")
            self.FC_SP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_batch_norm.save_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm")
            del self.FC_SP_16_256_batch_norm

            self.FC_MP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_batch_norm.save_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm")
            del self.FC_MP_16_256_batch_norm

            self.FC_MP_32_512_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_batch_norm.save_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm")
            del self.FC_MP_32_512_batch_norm

            self.VGG_2B_32_64_dropout_03.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_dropout_03.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_dropout_03.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_03")
            del self.VGG_2B_32_64_dropout_03

            self.VGG_3B_16_64_dropout_03.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_dropout_03.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_dropout_03.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_03")
            del self.VGG_3B_16_64_dropout_03

            self.VGG_3B_32_128_dropout_03.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_dropout_03.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_dropout_03.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_03")
            del self.VGG_3B_32_128_dropout_03

        # ================================================= batch norm, dropout 0.4 =================================================
        if "dropout_04" in models_to_train or "all" in models_to_train:
            print("\n\nbatch norm, dropout_04 models training:")
            self.FC_SP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_batch_norm.save_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm")
            del self.FC_SP_16_256_batch_norm

            self.FC_MP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_batch_norm.save_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm")
            del self.FC_MP_16_256_batch_norm

            self.FC_MP_32_512_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_batch_norm.save_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm")
            del self.FC_MP_32_512_batch_norm

            self.VGG_2B_32_64_dropout_04.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_dropout_04.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_dropout_04.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_04")
            del self.VGG_2B_32_64_dropout_04

            self.VGG_3B_16_64_dropout_04.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_dropout_04.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_dropout_04.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_04")
            del self.VGG_3B_16_64_dropout_04

            self.VGG_3B_32_128_dropout_04.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_dropout_04.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_dropout_04.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_04")
            del self.VGG_3B_32_128_dropout_04

        # ================================================= batch norm, dropout 0.5 =================================================
        if "dropout_05" in models_to_train or "all" in models_to_train:
            print("\n\ndropout_05 models training:")
            self.FC_SP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_batch_norm.save_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm")
            del self.FC_SP_16_256_batch_norm


            self.FC_MP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_batch_norm.save_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm")
            del self.FC_MP_16_256_batch_norm

            self.FC_MP_32_512_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_batch_norm.save_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm")
            del self.FC_MP_32_512_batch_norm

            self.VGG_2B_32_64_dropout_05.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_dropout_05.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_dropout_05.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_05")
            del self.VGG_2B_32_64_dropout_05

            self.VGG_3B_16_64_dropout_05.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_dropout_05.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_dropout_05.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_05")
            del self.VGG_3B_16_64_dropout_05

        self.VGG_3B_32_128_dropout_05.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.VGG_3B_32_128_dropout_05.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        self.VGG_3B_32_128_dropout_05.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_05")
        del self.VGG_3B_32_128_dropout_05

        # ====================================================== l1l2 0.0001 ======================================================
        if "l1l2_00001" in models_to_train or "all" in models_to_train:
            print("\n\nl1l2_00001 models training:")
            self.FC_SP_16_256_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l1l2_00001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_00001")
            del self.FC_SP_16_256_l1l2_00001

            self.FC_MP_16_256_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l1l2_00001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_00001")
            del self.FC_MP_16_256_l1l2_00001

            self.FC_MP_32_512_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l1l2_00001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_00001")
            del self.FC_MP_32_512_l1l2_00001

            self.VGG_2B_32_64_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l1l2_00001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_00001")
            del self.VGG_2B_32_64_l1l2_00001

            self.VGG_3B_16_64_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l1l2_00001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_00001")
            del self.VGG_3B_16_64_l1l2_00001

            self.VGG_3B_32_128_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l1l2_00001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_00001")
            del self.VGG_3B_32_128_l1l2_00001

        # ====================================================== l1l2 0.001 ======================================================
        if "l1l2_0001" in models_to_train or "all" in models_to_train:
            print("\n\nl1l2_0001 models training:")
            self.FC_SP_16_256_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l1l2_0001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_0001")
            del self.FC_SP_16_256_l1l2_0001

            self.FC_MP_16_256_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l1l2_0001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_0001")
            del self.FC_MP_16_256_l1l2_0001

            self.FC_MP_32_512_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l1l2_0001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_0001")
            del self.FC_MP_32_512_l1l2_0001

            self.VGG_2B_32_64_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l1l2_0001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_0001")
            del self.VGG_2B_32_64_l1l2_0001

            self.VGG_3B_16_64_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l1l2_0001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_0001")
            del self.VGG_3B_16_64_l1l2_0001

            self.VGG_3B_32_128_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l1l2_0001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_0001")
            del self.VGG_3B_32_128_l1l2_0001

        # ====================================================== l1l2 0.01 ======================================================
        if "l1l2_001" in models_to_train or "all" in models_to_train:
            print("\n\nl1l2_001 models training:")
            self.FC_SP_16_256_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l1l2_001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_001")
            del self.FC_SP_16_256_l1l2_001

            self.FC_MP_16_256_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l1l2_001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_001")
            del self.FC_MP_16_256_l1l2_001

            self.FC_MP_32_512_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l1l2_001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_001")
            del self.FC_MP_32_512_l1l2_001

            self.VGG_2B_32_64_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l1l2_001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_001")
            del self.VGG_2B_32_64_l1l2_001

            self.VGG_3B_16_64_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l1l2_001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_001")
            del self.VGG_3B_16_64_l1l2_001

            self.VGG_3B_32_128_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l1l2_001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_001")
            del self.VGG_3B_32_128_l1l2_001

        # ====================================================== l1 0.0001 ======================================================
        if "l1_00001" in models_to_train or "all" in models_to_train:
            print("\n\nl1_00001 models training:")
            self.FC_SP_16_256_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l1_00001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_00001")
            del self.FC_SP_16_256_l1_00001

            self.FC_MP_16_256_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l1_00001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_00001")
            del self.FC_MP_16_256_l1_00001

            self.FC_MP_32_512_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l1_00001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_00001")
            del self.FC_MP_32_512_l1_00001

            self.VGG_2B_32_64_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l1_00001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_00001")
            del self.VGG_2B_32_64_l1_00001

            self.VGG_3B_16_64_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l1_00001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_00001")
            del self.VGG_3B_16_64_l1_00001

            self.VGG_3B_32_128_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l1_00001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_00001")
            del self.VGG_3B_32_128_l1_00001

        # ====================================================== l1 0.001 ======================================================
        if "l1_0001" in models_to_train or "all" in models_to_train:
            print("\n\nl1_0001 models training:")
            self.FC_SP_16_256_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l1_0001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_0001")
            del self.FC_SP_16_256_l1_0001

            self.FC_MP_16_256_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l1_0001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_0001")
            del self.FC_MP_16_256_l1_0001

            self.FC_MP_32_512_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l1_0001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_0001")
            del self.FC_MP_32_512_l1_0001

            self.VGG_2B_32_64_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l1_0001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_0001")
            del self.VGG_2B_32_64_l1_0001

            self.VGG_3B_16_64_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l1_0001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_0001")
            del self.VGG_3B_16_64_l1_0001

            self.VGG_3B_32_128_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l1_0001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_0001")
            del self.VGG_3B_32_128_l1_0001

        # ====================================================== l1 0.01 ======================================================
        if "l1_001" in models_to_train or "all" in models_to_train:
            print("\n\nl1_001 models training:")
            self.FC_SP_16_256_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l1_001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_001")
            del self.FC_SP_16_256_l1_001

            self.FC_MP_16_256_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l1_001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_001")
            del self.FC_MP_16_256_l1_001

            self.FC_MP_32_512_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l1_001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_001")
            del self.FC_MP_32_512_l1_001

            self.VGG_2B_32_64_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l1_001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_001")
            del self.VGG_2B_32_64_l1_001

            self.VGG_3B_16_64_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l1_001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_001")
            del self.VGG_3B_16_64_l1_001

            self.VGG_3B_32_128_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l1_001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_001")
            del self.VGG_3B_32_128_l1_001

        # ====================================================== l2 0.0001 ======================================================
        if "l2_00001" in models_to_train or "all" in models_to_train:
            print("\n\nl2_00001 models training:")
            self.FC_SP_16_256_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l2_00001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_00001")
            del self.FC_SP_16_256_l2_00001

            self.FC_MP_16_256_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l2_00001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_00001")
            del self.FC_MP_16_256_l2_00001

            self.FC_MP_32_512_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l2_00001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_00001")
            del self.FC_MP_32_512_l2_00001

            self.VGG_2B_32_64_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l2_00001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_00001")
            del self.VGG_2B_32_64_l2_00001

            self.VGG_3B_16_64_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l2_00001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_00001")
            del self.VGG_3B_16_64_l2_00001

            self.VGG_3B_32_128_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l2_00001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_00001")
            del self.VGG_3B_32_128_l2_00001

        # ====================================================== l2 0.001 ======================================================
        if "l2_0001" in models_to_train or "all" in models_to_train:
            print("\n\nl2_0001 models training:")
            self.FC_SP_16_256_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l2_0001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_0001")
            del self.FC_SP_16_256_l2_0001

            self.FC_MP_16_256_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l2_0001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_0001")
            del self.FC_MP_16_256_l2_0001

            self.FC_MP_32_512_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l2_0001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_0001")
            del self.FC_MP_32_512_l2_0001

            self.VGG_2B_32_64_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l2_0001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_0001")
            del self.VGG_2B_32_64_l2_0001

            self.VGG_3B_16_64_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l2_0001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_0001")
            del self.VGG_3B_16_64_l2_0001

            self.VGG_3B_32_128_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l2_0001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_0001")
            del self.VGG_3B_32_128_l2_0001

        # ====================================================== l2 0.01 ======================================================
        if "l2_001" in models_to_train or "all" in models_to_train:
            print("\n\nl2_001 models training:")
            self.FC_SP_16_256_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_SP_16_256_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_SP_16_256_l2_001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_001")
            del self.FC_SP_16_256_l2_001

            self.FC_MP_16_256_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_16_256_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_16_256_l2_001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_001")
            del self.FC_MP_16_256_l2_001

            self.FC_MP_32_512_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.FC_MP_32_512_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.FC_MP_32_512_l2_001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_001")
            del self.FC_MP_32_512_l2_001

            self.VGG_2B_32_64_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_2B_32_64_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_2B_32_64_l2_001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_001")
            del self.VGG_2B_32_64_l2_001

            self.VGG_3B_16_64_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_16_64_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                    callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_16_64_l2_001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_001")
            del self.VGG_3B_16_64_l2_001

            self.VGG_3B_32_128_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.VGG_3B_32_128_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
            self.VGG_3B_32_128_l2_001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_001")
            del self.VGG_3B_32_128_l2_001

class PlotCallback(tfc.Callback):
    def __init__(self, validation=False):
        super().__init__()
        self.validation = validation
        self.loss_values = []
        
        if self.validation:
            self.val_loss_values = []
            self.on_test_batch_end = self.collect_validation_loss
    
    def on_train_batch_end(self, _, logs):
        self.loss_values.append(logs["loss"])

    def collect_validation_loss(self, _, logs):
        self.val_loss_values.append(logs["loss"])
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values)
        plt.title("title")
        plt.ylabel("loss")
        plt.xlabel("batch")
        
        if self.validation:
            plt.plot(np.linspace(0, len(self.loss_values), len(self.val_loss_values)), self.val_loss_values)
            plt.legend(["training", "validation"], loc="upper right")
        else:
            plt.legend(["training"], loc="upper right")

        plt.savefig("final_model_training_with_validation.png" if self.validation else "final_model_training.png")

class FinalModel():
    def __init__(self):
        self.model = tfm.Sequential([
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

        self.val_plot_callback = PlotCallback(True)
        self.plot_callback = PlotCallback(True)
    
    def train(self, x_train, y_train, epochs):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2, callbacks=self.plot_callback)
        self.model.save_weights("./models/final/model")

        self.plot_callback.plot()
    
    def train_with_validation(self, x_train, y_train, epochs):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, batch_size=32, verbose=2, callbacks=self.val_plot_callback)
        self.model.save_weights("./models/final/val_model")

        self.val_plot_callback.plot()
    
    def evaluate(self, x_test, y_test, validation=False):
        if validation:
            self.model.load_weights("./models/final/val_model").expect_partial()
        else:
            self.model.load_weights("./models/final/model").expect_partial()

        predictions = self.model.predict(x_test)
        predictions = (np.argmax(predictions, axis=1) == y_test).sum()
        accuracy = predictions / y_test.shape[0]
        print(f"final model accuracy:  {accuracy * 100:.2f} %")

        skm.plot_confusion_matrix(estimator=self.model, X=x_test, y_true=y_test, cmap="Blues", normalize="true", 
                                  ax=plt.subplots(figsize=(12, 12))[1])

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    (x_train, y_train), (x_test, y_test) = tfd.cifar10.load_data()
    x_train, x_test = np.mean(x_train, axis=3), np.mean(x_test, axis=3) # convert to grayscale
    x_train, x_test = x_train / 255, x_test / 255 # normalize to pixel values between 0 and 1
    x_train, x_test = tf.expand_dims(x_train, -1), tf.expand_dims(x_test, -1) # adding chanel dimension

    NUM_OF_CLASSES = 10
    NUM_OF_TEST_SAMPLES = len(y_test)

    y_train = tfu.to_categorical(y_train, num_classes=NUM_OF_CLASSES)
    y_test = np.array(y_test).reshape(-1)

    final_model = FinalModel()
    #final_model.train_with_validation(x_train, y_train, 16)
    #final_model.train(x_train, y_train, 16)
    final_model.evaluate()

