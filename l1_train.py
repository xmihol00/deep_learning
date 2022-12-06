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
x_train.shape, y_train.shape, x_test.shape, y_test.shape


full_conv_l1_reg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
full_conv_l1_reg.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
              callbacks=[tfc.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)])
full_conv_l1_reg.save_weights("./models/full_conv/full_conv_l1_reg")
del full_conv_l1_reg


full_conv_max_pool_l1_reg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
full_conv_max_pool_l1_reg.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                             callbacks=[tfc.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)])
full_conv_max_pool_l1_reg.save_weights("./models/full_conv_max_pool/full_conv_max_pool_l1_reg")
del full_conv_max_pool_l1_reg


VGG_inspired_less_kernels_l1_reg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
VGG_inspired_less_kernels_l1_reg.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                          callbacks=[tfc.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)])
VGG_inspired_less_kernels_l1_reg.save_weights("./models/VGG_inspired_less_kernels/VGG_inspired_less_kernels_l1_reg")
del VGG_inspired_less_kernels_l1_reg


VGG_inspired_more_kernels_l1_reg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
VGG_inspired_more_kernels_l1_reg.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                          callbacks=[tfc.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)])
VGG_inspired_more_kernels_l1_reg.save_weights("./models/VGG_inspired_more_kernels/VGG_inspired_more_kernels_l1_reg")
del VGG_inspired_more_kernels_l1_reg


VGG_inspired_3_dense_layers_l1_reg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
VGG_inspired_3_dense_layers_l1_reg.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                callbacks=[tfc.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)])
VGG_inspired_3_dense_layers_l1_reg.save_weights("./models/VGG_inspired_3_dense_layers/VGG_inspired_3_dense_layers_l1_reg")
del VGG_inspired_3_dense_layers_l1_reg
