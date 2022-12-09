import tensorflow.keras.datasets as tfd
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow as tf
import numpy as np
import random
from models import *

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
y_test = np.array(y_test).reshape(len(y_test))



FC_SP_16_256.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
FC_SP_16_256.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
              callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
FC_SP_16_256.save_weights("./models/FC_SP_16_256/FC_SP_16_256")
del FC_SP_16_256


FC_MP_16_256.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
FC_MP_16_256.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
              callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
FC_MP_16_256.save_weights("./models/FC_MP_16_256/FC_MP_16_256")
del FC_MP_16_256


FC_MP_32_512.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
FC_MP_32_512.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                             callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
FC_MP_32_512.save_weights("./models/FC_MP_32_512/FC_MP_32_512")
del FC_MP_32_512


VGG_2B_32_64.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
VGG_2B_32_64.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
VGG_2B_32_64.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64")
del VGG_2B_32_64


VGG_3B_16_64.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
VGG_3B_16_64.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                          callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
VGG_3B_16_64.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64")
del VGG_3B_16_64


VGG_3B_32_128.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
VGG_3B_32_128.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                          callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
VGG_3B_32_128.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128")
del VGG_3B_32_128
