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
import pickle
import skimage.util as sku
import PIL as pil
import copy
import sys

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
        if type(models_to_train) == str and models_to_train == "baseline":
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
            return

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
    
    def baseline_eval(self, x_test, y_test):
        NUM_OF_TEST_SAMPLES = y_test.shape[0]

        self.FC_SP_16_256.load_weights("./models/FC_SP_16_256/FC_SP_16_256").expect_partial()
        FC_SP_16_256_pred = self.FC_SP_16_256.predict(x_test)
        FC_SP_16_256_accuracy = (np.argmax(FC_SP_16_256_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"FC_SP_16_256 accuracy:  {FC_SP_16_256_accuracy * 100:.2f} %")
        del self.FC_SP_16_256

        self.FC_MP_16_256.load_weights("./models/FC_MP_16_256/FC_MP_16_256").expect_partial()
        FC_MP_16_256_pred = self.FC_MP_16_256.predict(x_test)
        FC_MP_16_256_accuracy = (np.argmax(FC_MP_16_256_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"FC_MP_16_256 accuracy:  {FC_MP_16_256_accuracy * 100:.2f} %")
        del self.FC_MP_16_256


        self.FC_MP_32_512.load_weights("./models/FC_MP_32_512/FC_MP_32_512").expect_partial()
        FC_MP_32_512_pred = self.FC_MP_32_512.predict(x_test)
        FC_MP_32_512_accuracy = (np.argmax(FC_MP_32_512_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"FC_MP_32_512 accuracy:  {FC_MP_32_512_accuracy * 100:.2f} %")
        del self.FC_MP_32_512


        self.VGG_2B_32_64.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64").expect_partial()
        VGG_2B_32_64_pred = self.VGG_2B_32_64.predict(x_test)
        VGG_2B_32_64_accuracy = (np.argmax(VGG_2B_32_64_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"VGG_2B_32_64 accuracy:  {VGG_2B_32_64_accuracy * 100:.2f} %")
        del self.VGG_2B_32_64


        self.VGG_3B_16_64.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64").expect_partial()
        VGG_3B_16_64_pred = self.VGG_3B_16_64.predict(x_test)
        VGG_3B_16_64_accuracy = (np.argmax(VGG_3B_16_64_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"VGG_3B_16_64 accuracy:  {VGG_3B_16_64_accuracy * 100:.2f} %")
        del self.VGG_3B_16_64


        self.VGG_3B_32_128.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128").expect_partial()
        VGG_3B_32_128_pred = self.VGG_3B_32_128.predict(x_test)
        VGG_3B_32_128_accuracy = (np.argmax(VGG_3B_32_128_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"VGG_3B_32_128 accuracy: {VGG_3B_32_128_accuracy * 100:.2f} %")
        del self.VGG_3B_32_128
    
    def run(self, mode, models_to_run, x_train, y_train, x_test=None, y_test=None):
        if mode == "all" or mode == "train":
            self.train(models_to_run, x_train, y_train)
        
        if type(models_to_run) == str and models_to_run == "baseline" and (mode == "all" or mode == "eval"):
            self.baseline_eval(x_test, y_test)

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
        plt.title("Training and validation loss" if self.validation else "Training loss")
        plt.ylabel("loss")
        plt.xlabel("batch")
        
        filename = ""
        if self.validation:
            plt.plot(np.linspace(0, len(self.loss_values), len(self.val_loss_values)), self.val_loss_values)
            plt.legend(["training", "validation"], loc="upper right")
            filename = "final_model_training_with_validation.png"
        else:
            filename = "final_model_training.png"

        plt.savefig(filename)
        plt.show()
        plt.close()

class FinalModel():
    def __init__(self, model_type=None):
        self.model_type = model_type
        self.val_plot_callback = PlotCallback(True)
        self.plot_callback = PlotCallback(False)

        if self.model_type == "l2_bn":
            self.model = tfm.Sequential([
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Flatten(),
                tfl.Dense(256, activation="relu", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Dense(10, activation="softmax")
            ])
        elif self.model_type == "avg_pool":
            self.model = tfm.Sequential([
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                tfl.BatchNormalization(),
                tfl.AveragePooling2D(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
                tfl.BatchNormalization(),
                tfl.AveragePooling2D(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
                tfl.BatchNormalization(),
                tfl.AveragePooling2D(),
                tfl.Flatten(),
                tfl.Dropout(0.5),
                tfl.Dense(256, activation="relu"),
                tfl.Dropout(0.5),
                tfl.Dense(10, activation="softmax")
            ])
        else:
            self.model_type = "final" if self.model_type == None else self.model_type
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
        
        self.model.build((None, 32, 32, 1))
        self.weights = self.model.get_weights()
    
    def train(self, x_train, y_train, epochs):
        self.model.set_weights(self.weights)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2, callbacks=self.plot_callback)
        self.model.save_weights(f"./models/final/{self.model_type}_model")

        self.plot_callback.plot()
    
    def train_with_validation(self, x_train, y_train, epochs):
        self.model.set_weights(self.weights)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, batch_size=32, verbose=2, callbacks=self.val_plot_callback)
        self.model.save_weights(f"./models/final/val_{self.model_type}_model")

        self.val_plot_callback.plot()
    
    def evaluate(self, x_test, y_test, dataset="Test"):
        self.model.load_weights(f"./models/final/{self.model_type}_model").expect_partial()
        predictions = self.final_model.predict(x_test)
        predictions = np.argmax(predictions, axis=1)        
        accuracy = (predictions == y_test).sum() / y_test.shape[0]
        print(f"{self.model_type} model accuracy:  {accuracy * 100:.2f} %")

        confusion_matrix = tf.math.confusion_matrix(y_test, predictions).numpy()
        confusion_matrix = skm.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["airplane", 
                                                                                                         "automobile", 
                                                                                                         "bird", 
                                                                                                         "cat", 
                                                                                                         "deer", 
                                                                                                         "dog", 
                                                                                                         "frog", 
                                                                                                         "horse", 
                                                                                                         "ship", 
                                                                                                         "truck"])
        axis = plt.subplots(figsize=(12, 12))[1]
        plt.title(f"{dataset} data set confusion matrix")
        confusion_matrix.plot(cmap="Blues", ax=axis)
        plt.savefig(f"{self.model_type}_confusion_matrix.png")
        plt.show()
    
    def evaluate_with_validation(self, x_test, y_test):
        self.final_model.load_weights(f"./models/final/val_{self.model_type}_model").expect_partial()
        predictions = self.final_model.predict(x_test)
        predictions = np.argmax(predictions, axis=1)        
        accuracy = (predictions == y_test).sum() / y_test.shape[0]
        print(f"{self.model_type} model accuracy trained with validation data set:  {accuracy * 100:.2f} %")

def augmentation_plot(x_test, x_perturb):
    for test_image, perturb_image in zip(x_test, x_perturb):
        figure, axis = plt.subplots(3, 3)
        figure.set_size_inches(14, 9)

        contrast_image = np.asarray(pil.ImageEnhance.Brightness(pil.Image.fromarray(test_image, mode="RGB")).enhance(1.15)) / 255
        test_image = test_image / 255
        perturb_image = perturb_image / 255
        salt_and_pepper_image = sku.random_noise(test_image, mode="s&p", amount=0.0175)
        gaussian_image = sku.random_noise(test_image, mode="gaussian", var=0.005)
        perturb_test_difference = perturb_image - test_image
        salt_and_pepper_difference = salt_and_pepper_image - test_image
        gaussian_difference = gaussian_image - test_image
        contrast_difference = contrast_image - test_image

        axis[0, 0].imshow(test_image)
        axis[0, 0].set_title("Test image")
        
        axis[0, 1].imshow(perturb_image)
        axis[0, 1].set_title("Perturbed image")

        axis[0, 2].imshow(perturb_test_difference)
        axis[0, 2].set_title(f"Difference between perturbed and test image (mean: {np.mean(perturb_test_difference):.2E})")

        axis[1, 0].imshow(salt_and_pepper_image)
        axis[1, 0].set_title("Test image with salt and pepper noise")

        axis[1, 1].imshow(gaussian_image)
        axis[1, 1].set_title("Test image with gaussian noise")

        axis[1, 2].imshow(contrast_image)
        axis[1, 2].set_title("Test image with decreased contrast")

        axis[2, 0].imshow(salt_and_pepper_difference)
        axis[2, 0].set_title("Difference salt and peper")

        axis[2, 1].imshow(gaussian_difference)
        axis[2, 1].set_title("Difference gaussian")

        axis[2, 2].imshow(contrast_difference)
        axis[2, 2].set_title("Difference decreased contrast")
        
        plt.show()

def augment_data_set(x_dataset, y_dataset):
    x_salt_pepper_set = np.zeros_like(x_dataset)
    x_gaussian_set = np.zeros_like(x_dataset)

    y_salt_pepper_set = np.zeros_like(y_dataset)
    y_gaussian_set = np.zeros_like(y_dataset)

    for i, image in enumerate(x_dataset):
        image = image.numpy()

        x_salt_pepper_set[i] = sku.random_noise(image, mode="s&p", amount=0.0175)
        x_gaussian_set[i] = sku.random_noise(image, mode="gaussian", var=0.005)

        y_salt_pepper_set[i] = y_dataset[i]
        y_gaussian_set[i] = y_dataset[i]

    x_concat = np.concatenate((x_dataset, x_salt_pepper_set, x_gaussian_set), 0)
    y_concat = np.concatenate((y_dataset, y_salt_pepper_set, y_gaussian_set), 0)
    shuffel = np.random.choice(y_concat.shape[0], y_concat.shape[0], replace=False)
    return x_concat[shuffel], y_concat[shuffel]

def run_final_models(mode, models_to_run, x_train, y_train, x_test, y_test, x_perturb, y_perturb, epochs=16):
    final_model = None
    l2_bn_model = None
    avg_pool_model = None
    augment_model = None
    
    if "all" in models_to_run:
        final_model = FinalModel()
        l2_bn_model = FinalModel("l2_bn")
        avg_pool_model = FinalModel("avg_pool")
        augment_model = FinalModel("augment")
    else:
        if "final" in models_to_run:
            final_model = FinalModel()
        if "augment" in models_to_run:
            augment_model = FinalModel("augment")
        if "l2" in models_to_run:
            l2_bn_model = FinalModel("l2_bn")
        if "avg_pool" in models_to_run:
            avg_pool_model = FinalModel("avg_pool")
    
    if mode == "all":
        if final_model:
            final_model.train_with_validation(x_train, y_train, epochs)
            final_model.evaluate_with_validation(x_test, y_test)
            final_model.train(x_train, y_train, epochs)
            final_model.evaluate(x_test, y_test)
            final_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if l2_bn_model:
            l2_bn_model.train(x_train, y_train, epochs)
            l2_bn_model.evaluate(x_test, y_test)
            l2_bn_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if avg_pool_model:
            avg_pool_model.train(x_train, y_train, epochs)
            avg_pool_model.evaluate(x_test, y_test)
            avg_pool_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if augment_model:
            x_train, y_train = augment_data_set(x_train, y_train)

            augment_model.train(x_train, y_train, epochs)
            augment_model.evaluate(x_test, y_test)
            augment_model.evaluate(x_perturb, y_perturb, "Perturbed")

    elif mode == "train":
        if final_model:
            final_model.train(x_train, y_train, epochs)
        
        if l2_bn_model:
            l2_bn_model.train(x_train, y_train, epochs)
        
        if avg_pool_model:
            avg_pool_model.train(x_train, y_train, epochs)
        
        if augment_model:
            x_train, y_train = augment_data_set(x_train, y_train)

            augment_model.train(x_train, y_train, epochs)

    elif mode == "val_train":
        if final_model:
            final_model.train_with_validation(x_train, y_train, epochs)

    elif mode == "eval":
        if final_model:
            final_model.evaluate(x_test, y_test)
            final_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if l2_bn_model:
            l2_bn_model.evaluate(x_test, y_test)
            l2_bn_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if avg_pool_model:
            avg_pool_model.evaluate(x_test, y_test)
            avg_pool_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if augment_model:
            x_train, y_train = augment_data_set(x_train, y_train)

            augment_model.evaluate(x_test, y_test)
            augment_model.evaluate(x_perturb, y_perturb, "Perturbed")

    elif mode == "val_eval":
        if final_model:
            final_model.evaluate_with_validation(x_test, y_test)

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    baseline = False
    regularization = False
    final = False
    plot = False
    models_to_run = ["all"]
    mode = "all"
    
    if len(sys.argv) == 1:
        baseline = True
        final = True
        regularization = True
    elif len(sys.argv) > 1:
        if sys.argv[1].lower() == "plot":
            plot = True
        elif sys.argv[1].lower() == "baseline":
            baseline = True
        elif sys.argv[1].lower() == "reg" or sys.argv[1].lower() == "regularization":
            regularization = True
        elif sys.argv[1].lower() == "final":
            final = True
        
        index = 2
        if len(sys.argv) > 2:
            if sys.argv[1].lower() == "train":
                mode = "train"
                index = 3
            elif sys.argv[1].lower() == "eval":
                mode = "eval"
                index = 3
            elif sys.argv[1].lower() == "val_eval":
                mode = "val_eval"
                index = 3
        
        if len(sys.argv) > index:
            models_to_run = sys.argv[index:]        

    (x_train_RGB, y_train), (x_test_RGB, y_test) = tfd.cifar10.load_data()
    x_train, x_test = np.mean(x_train_RGB, axis=3), np.mean(x_test_RGB, axis=3) # convert to grayscale
    x_train, x_test = x_train / 255, x_test / 255 # normalize to pixel values between 0 and 1
    x_train, x_test = tf.expand_dims(x_train, -1), tf.expand_dims(x_test, -1) # adding chanel dimension

    dict = pickle.load(open("cifar10_perturb_test.pickle", "rb"))
    x_perturb_RGB, y_perturb = dict["x_perturb"], dict["y_perturb"]
    x_perturb = np.mean(x_perturb_RGB, axis=3) # convert to grayscale
    x_perturb = x_perturb / 255 # normalize to pixel values between 0 and 1
    x_perturb = np.expand_dims(x_perturb, -1) # adding chanel dimension

    if plot:
        augmentation_plot(x_test_RGB, x_perturb_RGB)
        exit(0)
    del x_perturb_RGB, x_train_RGB, x_test_RGB # RGB images are not needed anymore

    NUM_OF_CLASSES = 10 # CIFAR-10
    y_train = tfu.to_categorical(y_train, num_classes=NUM_OF_CLASSES) # one-hot encoding of train labels
    y_test = np.array(y_test).reshape(-1)       # test labels to 1 dimensional array
    y_perturb = np.array(y_perturb).reshape(-1) # perturb labels to 1 dimensional array

    models = None
    if baseline or regularization:
        models = Models()

    if baseline:
        models.run(mode, "baseline", x_train, y_train, x_test, y_test)
    
    if regularization:
        models.run(mode, models_to_run, x_train, y_train)
    
    if final:
        EPOCHS = 16 # number of epochs discovered as the best when training with early stopping
        run_final_models(mode, models_to_run, x_train, y_train, x_test, y_test, x_perturb, y_perturb, EPOCHS)
