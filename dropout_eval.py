import tensorflow.keras.datasets as tfd
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow as tf
import numpy as np
from models import *

(x_train, y_train), (x_test, y_test) = tfd.cifar10.load_data()
x_train, x_test = np.mean(x_train, axis=3), np.mean(x_test, axis=3) # convert to grayscale
x_train, x_test = x_train / 255, x_test / 255 # normalize to pixel values between 0 and 1
x_train, x_test = tf.expand_dims(x_train, -1), tf.expand_dims(x_test, -1) # adding chanel dimension

NUM_OF_CLASSES = 10
NUM_OF_TEST_SAMPLES = len(y_test)

y_train = tfu.to_categorical(y_train, num_classes=NUM_OF_CLASSES)
y_test = np.array(y_test).reshape(len(y_test))
x_train.shape, y_train.shape, x_test.shape, y_test.shape


FC_SP_16_256_batch_norm.load_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm").expect_partial()
FC_SP_16_256_batch_norm_pred = FC_SP_16_256_batch_norm.predict(x_test)
FC_SP_16_256_batch_norm_accuracy = (np.argmax(FC_SP_16_256_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"FC_SP_16_256_batch_norm accuracy:  {FC_SP_16_256_batch_norm_accuracy * 100:.2f} %")
del FC_SP_16_256_batch_norm


FC_MP_16_256_batch_norm.load_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm").expect_partial()
FC_MP_16_256_batch_norm_pred = FC_MP_16_256_batch_norm.predict(x_test)
FC_MP_16_256_batch_norm_accuracy = (np.argmax(FC_MP_16_256_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"FC_MP_16_256_batch_norm accuracy:  {FC_MP_16_256_batch_norm_accuracy * 100:.2f} %")
del FC_MP_16_256_batch_norm


FC_MP_32_512_batch_norm.load_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm").expect_partial()
FC_MP_32_512_batch_norm_pred = FC_MP_32_512_batch_norm.predict(x_test)
FC_MP_32_512_batch_norm_accuracy = (np.argmax(FC_MP_32_512_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"FC_MP_32_512_batch_norm accuracy:  {FC_MP_32_512_batch_norm_accuracy * 100:.2f} %")
del FC_MP_32_512_batch_norm


VGG_2B_32_64_dropout_05.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_05").expect_partial()
VGG_2B_32_64_dropout_05_pred = VGG_2B_32_64_dropout_05.predict(x_test)
VGG_2B_32_64_dropout_05_accuracy = (np.argmax(VGG_2B_32_64_dropout_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"VGG_2B_32_64_dropout_05 accuracy:  {VGG_2B_32_64_dropout_05_accuracy * 100:.2f} %")
del VGG_2B_32_64_dropout_05


VGG_3B_16_64_dropout_05.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_05").expect_partial()
VGG_3B_16_64_dropout_05_pred = VGG_3B_16_64_dropout_05.predict(x_test)
VGG_3B_16_64_dropout_05_accuracy = (np.argmax(VGG_3B_16_64_dropout_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"VGG_3B_16_64_dropout_05 accuracy:  {VGG_3B_16_64_dropout_05_accuracy * 100:.2f} %")
del VGG_3B_16_64_dropout_05


VGG_3B_32_128_dropout_05.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_05").expect_partial()
VGG_3B_32_128_dropout_05_pred = VGG_3B_32_128_dropout_05.predict(x_test)
VGG_3B_32_128_dropout_05_accuracy = (np.argmax(VGG_3B_32_128_dropout_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"VGG_3B_32_128_dropout_05 accuracy: {VGG_3B_32_128_dropout_05_accuracy * 100:.2f} %")
del VGG_3B_32_128_dropout_05

