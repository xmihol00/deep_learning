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


full_conv_batch_norm.load_weights("./models/full_conv/full_conv_batch_norm").expect_partial()
full_conv_batch_norm_pred = full_conv_batch_norm.predict(x_test)
full_conv_batch_norm_accuracy = (np.argmax(full_conv_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"full_conv_accuracy:                             {full_conv_batch_norm_accuracy * 100:.2f} %")
del full_conv


model_full_conv_max_pool_batch_norm.load_weights("./models/model_full_conv_max_pool/model_full_conv_max_pool_batch_norm").expect_partial()
model_full_conv_max_pool_batch_norm_pred = model_full_conv_max_pool_batch_norm.predict(x_test)
model_full_conv_max_pool_batch_norm_accuracy = (np.argmax(model_full_conv_max_pool_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"model_full_conv_max_pool_batch_norm_accuracy:   {model_full_conv_max_pool_batch_norm_accuracy * 100:.2f} %")
del model_full_conv_max_pool_batch_norm


VGG_inspired_less_kernels_droput_05.load_weights("./models/VGG_inspired_less_kernels/VGG_inspired_less_kernels_droput_05").expect_partial()
VGG_inspired_less_kernels_droput_05_pred = VGG_inspired_less_kernels_droput_05.predict(x_test)
VGG_inspired_less_kernels_droput_05_accuracy = (np.argmax(VGG_inspired_less_kernels_droput_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"VGG_inspired_less_kernels_droput_05_accuracy:   {VGG_inspired_less_kernels_droput_05_accuracy * 100:.2f} %")
del VGG_inspired_less_kernels_droput_05


VGG_inspired_more_kernels_droput_05.load_weights("./models/VGG_inspired_more_kernels/VGG_inspired_more_kernels_droput_05").expect_partial()
VGG_inspired_more_kernels_droput_05_pred = VGG_inspired_more_kernels_droput_05.predict(x_test)
VGG_inspired_more_kernels_droput_05_accuracy = (np.argmax(VGG_inspired_more_kernels_droput_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"VGG_inspired_more_kernels_droput_05_accuracy:   {VGG_inspired_more_kernels_droput_05_accuracy * 100:.2f} %")
del VGG_inspired_more_kernels_droput_05


VGG_inspired_3_dense_layers_droput_05.load_weights("./models/VGG_inspired_3_dense_layers/VGG_inspired_3_dense_layers_droput_05").expect_partial()
VGG_inspired_3_dense_layers_droput_05_pred = VGG_inspired_3_dense_layers_droput_05.predict(x_test)
VGG_inspired_3_dense_layers_droput_05_accuracy = (np.argmax(VGG_inspired_3_dense_layers_droput_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
print(f"VGG_inspired_3_dense_layers_droput_05_accuracy: {VGG_inspired_3_dense_layers_droput_05_accuracy * 100:.2f} %")
#del VGG_inspired_3_dense_layers_droput_05
