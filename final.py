# ========================================================
# Author: David Mihola (david.mihola@student.tugraz.at)
# Matrikelnummer: 12211951
# Date: 21. 12. 2022
# ========================================================

###################################### How to run the script and reproduce results from the report ########################################
# 1, Runing the script simply as `python assigment3_mihola.py` will train all the discussed models in the report and evaluate all 
#    the necessary ones. The results should match the results written in the report.
# 2, Each group of models or a single model can be also run separately. This can be done by supplying command line arguments to the 
#    script when executing it. The command line arguments are separated into 3 types:
#      * section of the report with possible values - 'baseline' for the 1st section, 'regularization' or 'reg' fro the second section,
#                                                     'final' for the  3rd and 4th section and 'plot' to plot the inspection of the 
#                                                     perturbed data set,
#      * mode of running - 'train' to train the selected models (in case of section 1 and 2 with validation data set), 'eval' to evaluate 
#                          the selected models, 'val_train' to train the final model with validation dat set, 'val_eval' to 
#                          evaluate the final model trained with validation data set, 'all' to perform all listed modes,
#      * model or group of models - 'final' the final model, 'avg_pool' the modified final model with average pooling layers,
#                                   'l2_bn' the modified final model with l2 regularization in convolutional layers and with dropout 
#                                   replaced by l2 regularization, 'l2_bn_dropout' modified final model with l2 regularization in 
#                                   convolutional and fully connected layers and with 0.25 dropout, 'augment' the final model trained on
#                                   the augmented training data set, 'batch_norm' the Darknet-53 inspired models only with batch
#                                   normalization layers, 'dropout_03' the models with dropout rate 0.3, 'dropout_04' the models with 
#                                   dropout rate 0.4, 'dropout_05' the models with dropout rate 0.5, 'l1_00001' the models with l1
#                                   regularization of 0.0001, 'l1_0001' the models with l1 regularization of 0.001, 'l1_001' the models 
#                                   with l1 regularization of 0.01, 'l2_00001' the models with l2 regularization of 0.0001, 'l2_0001' 
#                                   the models with l2 regularization of 0.001, 'l2_001' the models with l2 regularization of 0.01,
#                                   'l1l2_00001' the models with l1 and l2 regularization of 0.0001, 'l1l2_0001' the models with l1 and 
#                                   l2 regularization of 0.001, 'l1l2_001' the models with l1 and l2 regularization of 0.01.
#    The script is expected to be called `python assigment3_mihola.py [section of the report] [mode of runing] [model/models]*`.
#    Examples:
#      * `python assigment3_mihola.py baseline train` to obtain the results for the Table 1 in the report,
#      * `python assigment3_mihola.py reg train batch_norm dropout_03 dropout_04 dropout_05 l1_001 l1_0001 l1_00001` to obtain the 
#         results for the Table 2 in the report,
#      * `python assigment3_mihola.py reg train l2_001 l2_0001 l2_00001 l1l2_001 l1l2_0001 l1l2_00001` to obtain the results for the 
#         Table 3 in the report,
#      * `python assigment3_mihola.py final all final` to obtain the Figure 1 and Figure 2 in the report,
#      * `python assigment3_mihola.py final all all` to obtain the results for Table 5 in the report.
#      * `python assigment3_mihola.py plot` to plot my salt and pepper and gaussian augmentation in comparrison to the perturbed data set.
###########################################################################################################################################

import tensorflow.keras.datasets as tfd
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow.keras.regularizers as tfr
import tensorflow.keras.initializers as tfi
import tensorflow_addons as tfa
import sklearn.metrics as skm
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import skimage.util as sku
import PIL as pil
import sys

RANDOM_SEED = 1

class Models():  
    def assign_models(self):
        # ====================================================== baseline ======================================================
        self.FC_SP_16_256 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ===================================================== batch norm =====================================================
        self.FC_SP_16_256_batch_norm = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_batch_norm = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_batch_norm = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        # ==================================================== dropout 0.5 ====================================================
        self.VGG_2B_32_64_dropout_05 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.5),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.5),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.5),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_dropout_05 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.5),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.5),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_dropout_05 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.5),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.5),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== dropout 0.4 ====================================================
        self.VGG_2B_32_64_dropout_04 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.4),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.4),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.4),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_dropout_04 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.4),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.4),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_dropout_04 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.4),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.4),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== dropout 0.3 ====================================================
        self.VGG_2B_32_64_dropout_03 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.3),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.3),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.3),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_dropout_03 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.3),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.3),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_dropout_03 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.BatchNormalization(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dropout(0.3),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
            tfl.Dropout(0.3),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1 0.0001 ====================================================
        self.FC_SP_16_256_l1_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l1_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l1_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l1_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.0001)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l1_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l1_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1 0.001 ====================================================
        self.FC_SP_16_256_l1_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l1_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l1_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l1_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.001)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l1_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l1_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1 0.01 ====================================================
        self.FC_SP_16_256_l1_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l1_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l1_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l1_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.01)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l1_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l1_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l2 0.0001 ====================================================
        self.FC_SP_16_256_l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.0001)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1 0.001 ====================================================
        self.FC_SP_16_256_l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.001)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1 0.01 ====================================================
        self.FC_SP_16_256_l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.01)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1(0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1(0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])


        # ==================================================== l1l2 0.0001 ====================================================
        self.FC_SP_16_256_l1l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l1l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l1l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l1l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l1l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l1l2_00001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.0001, 0.0001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1l2 0.001 ====================================================
        self.FC_SP_16_256_l1l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l1l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l1l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l1l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l1l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l1l2_0001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.001, 0.001)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        # ==================================================== l1l2 0.01 ====================================================
        self.FC_SP_16_256_l1l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])                                                                               
        ])

        self.FC_MP_16_256_l1l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.FC_MP_32_512_l1l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=10, kernel_size=(1, 1), activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
            tfl.Reshape([10])
        ])

        self.VGG_2B_32_64_l1l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_16_64_l1l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(128, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
        ])

        self.VGG_3B_32_128_l1l2_001 = tfm.Sequential([
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L1L2(0.01, 0.01)),
            tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
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
        
        # ==================================================== batch norm. ====================================================
        if "batch_norm" in models_to_train or "all" in models_to_train:
            print("\n\nbatch norm models training:")
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

        # ==================================================== dropout 0.3 ====================================================
        if "dropout_03" in models_to_train or "all" in models_to_train:
            print("\n\ndropout_03 models training:")
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

        # ==================================================== dropout 0.4 ====================================================
        if "dropout_04" in models_to_train or "all" in models_to_train:
            print("\n\ndropout_04 models training:")
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

        # ==================================================== dropout 0.5 ====================================================
        if "dropout_05" in models_to_train or "all" in models_to_train:
            print("\n\ndropout_05 models training:")
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
        print("\n\nbaseline model evaluation:")
        NUM_OF_TEST_SAMPLES = y_test.shape[0]

        self.FC_SP_16_256.load_weights("./models/FC_SP_16_256/FC_SP_16_256").expect_partial()
        FC_SP_16_256_pred = self.FC_SP_16_256.predict(x_test, verbose=2)
        FC_SP_16_256_accuracy = (np.argmax(FC_SP_16_256_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"FC_SP_16_256 accuracy:  {FC_SP_16_256_accuracy * 100:.2f} %")
        del self.FC_SP_16_256

        self.FC_MP_16_256.load_weights("./models/FC_MP_16_256/FC_MP_16_256").expect_partial()
        FC_MP_16_256_pred = self.FC_MP_16_256.predict(x_test, verbose=2)
        FC_MP_16_256_accuracy = (np.argmax(FC_MP_16_256_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"FC_MP_16_256 accuracy:  {FC_MP_16_256_accuracy * 100:.2f} %")
        del self.FC_MP_16_256


        self.FC_MP_32_512.load_weights("./models/FC_MP_32_512/FC_MP_32_512").expect_partial()
        FC_MP_32_512_pred = self.FC_MP_32_512.predict(x_test, verbose=2)
        FC_MP_32_512_accuracy = (np.argmax(FC_MP_32_512_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"FC_MP_32_512 accuracy:  {FC_MP_32_512_accuracy * 100:.2f} %")
        del self.FC_MP_32_512


        self.VGG_2B_32_64.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64").expect_partial()
        VGG_2B_32_64_pred = self.VGG_2B_32_64.predict(x_test, verbose=2)
        VGG_2B_32_64_accuracy = (np.argmax(VGG_2B_32_64_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"VGG_2B_32_64 accuracy:  {VGG_2B_32_64_accuracy * 100:.2f} %")
        del self.VGG_2B_32_64


        self.VGG_3B_16_64.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64").expect_partial()
        VGG_3B_16_64_pred = self.VGG_3B_16_64.predict(x_test, verbose=2)
        VGG_3B_16_64_accuracy = (np.argmax(VGG_3B_16_64_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"VGG_3B_16_64 accuracy:  {VGG_3B_16_64_accuracy * 100:.2f} %")
        del self.VGG_3B_16_64


        self.VGG_3B_32_128.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128").expect_partial()
        VGG_3B_32_128_pred = self.VGG_3B_32_128.predict(x_test, verbose=2)
        VGG_3B_32_128_accuracy = (np.argmax(VGG_3B_32_128_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
        print(f"VGG_3B_32_128 accuracy: {VGG_3B_32_128_accuracy * 100:.2f} %")
        del self.VGG_3B_32_128
    
    def run(self, mode, models_to_run, x_train, y_train, x_test=None, y_test=None):
        if mode == "all" or mode == "train":
            self.assign_models()
            self.train(models_to_run, x_train, y_train)
        
        if type(models_to_run) == str and models_to_run == "baseline" and (mode == "all" or mode == "eval"):
            self.assign_models()
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
        plt.close()

class FinalModel():
    def __init__(self, model_type=None):
        self.model_type = model_type
        self.val_plot_callback = PlotCallback(True)
        self.plot_callback = PlotCallback(False)

        if self.model_type == "l2_bn":
            self.model = tfm.Sequential([
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Flatten(),
                tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
            ])
        elif self.model_type == "l2_bn_dropout":
            self.model = tfm.Sequential([
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same", kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Flatten(),
                tfl.Dropout(0.25),
                tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), kernel_regularizer=tfr.L2(0.0001)),
                tfl.BatchNormalization(),
                tfl.Dropout(0.25),
                tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
            ])
        elif self.model_type == "avg_pool":
            self.model = tfm.Sequential([
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.AveragePooling2D(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.AveragePooling2D(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.AveragePooling2D(),
                tfl.Flatten(),
                tfl.Dropout(0.4),
                tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
                tfl.Dropout(0.4),
                tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
            ])
        else:
            self.model_type = "final" if self.model_type == None else self.model_type
            self.model = tfm.Sequential([
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED), padding="same"),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(),
                tfl.Flatten(),
                tfl.Dropout(0.4),
                tfl.Dense(256, activation="relu", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED)),
                tfl.Dropout(0.4),
                tfl.Dense(10, activation="softmax", kernel_initializer=tfi.GlorotNormal(RANDOM_SEED))
            ])
        
        self.model.build((None, 32, 32, 1))
        self.weights = self.model.get_weights()
    
    def train(self, x_train, y_train, epochs):
        print(f"\n\n{self.model_type} model training:")

        self.model.set_weights(self.weights)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2, callbacks=self.plot_callback)
        self.model.save_weights(f"./models/final/{self.model_type}_model")

        self.plot_callback.plot()
    
    def train_with_validation(self, x_train, y_train, epochs):
        print(f"\n\n{self.model_type} model training with validation:")

        self.model.set_weights(self.weights)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, batch_size=32, verbose=2, callbacks=self.val_plot_callback)
        self.model.save_weights(f"./models/final/val_{self.model_type}_model")

        self.val_plot_callback.plot()
    
    def evaluate(self, x_test, y_test, dataset="Test"):
        print(f"\n\n{dataset} data set {self.model_type} model evaluation:")

        self.model.load_weights(f"./models/final/{self.model_type}_model").expect_partial()
        predictions = self.model.predict(x_test, verbose=2)
        predictions = np.argmax(predictions, axis=1)        
        accuracy = (predictions == y_test).sum() / y_test.shape[0]
        print(f"{dataset} data set {self.model_type} model accuracy:  {accuracy * 100:.2f} %")

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
        plt.savefig(f"{self.model_type}_{dataset.lower()}_confusion_matrix.png")
        plt.close()
    
    def evaluate_with_validation(self, x_test, y_test):
        print(f"\n\n{self.model_type} model evaluation with validation:")

        self.model.load_weights(f"./models/final/val_{self.model_type}_model").expect_partial()
        predictions = self.model.predict(x_test, verbose=2)
        predictions = np.argmax(predictions, axis=1)        
        accuracy = (predictions == y_test).sum() / y_test.shape[0]
        print(f"{self.model_type} model accuracy trained with validation data set:  {accuracy * 100:.2f} %")

def augmentation_plot(x_test, x_perturb):
    for test_image, perturb_image in zip(x_test, x_perturb):
        figure, axis = plt.subplots(3, 3)
        figure.set_size_inches(14, 9)

        contrast_image = np.asarray(pil.ImageEnhance.Contrast(pil.ImageEnhance.Brightness(pil.Image.fromarray(test_image, mode="RGB")).enhance(1.15)).enhance(1.1)) / 255
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
        axis[0, 2].set_title(f"Difference between perturbed and test image")

        axis[1, 0].imshow(salt_and_pepper_image)
        axis[1, 0].set_title("Test image with salt and pepper noise")

        axis[1, 1].imshow(gaussian_image)
        axis[1, 1].set_title("Test image with gaussian noise")

        axis[1, 2].imshow(contrast_image)
        axis[1, 2].set_title("Test image with increased brightness and contrast")

        axis[2, 0].imshow(salt_and_pepper_difference)
        axis[2, 0].set_title("Difference salt and peper")

        axis[2, 1].imshow(gaussian_difference)
        axis[2, 1].set_title("Difference gaussian")

        axis[2, 2].imshow(contrast_difference)
        axis[2, 2].set_title("Difference decreased contrast")
        
        plt.show()

def salt_and_pepper_and_gaussian_augment(x_dataset, y_dataset):
    np.random.seed(RANDOM_SEED)

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

def rotation_and_colors_augment(x_dataset, y_dataset):
    np.random.seed(RANDOM_SEED)

    x_rotation_set = np.zeros_like(x_dataset)
    x_colors_set = np.zeros_like(x_dataset)

    y_rotation_set = np.zeros_like(y_dataset)
    y_colors_set = np.zeros_like(y_dataset)

    for i, image in enumerate(x_dataset):
        image = image.numpy()

        x_rotation_set[i] = tfa.image.rotate(image, (np.random.randn() + 1) * np.pi / 6) # randomly rotate at maximum by 30 degrees
        x_colors_set[i] = np.asarray(
                              pil.ImageEnhance.Contrast(
                                  pil.ImageEnhance.Brightness(
                                      pil.Image.fromarray(image.reshape(32, 32), mode="L")
                                  )
                                  .enhance(1.15)
                              )
                              .enhance(1.1)
                          ).reshape(32, 32, 1) / 255

        y_rotation_set[i] = y_dataset[i]
        y_colors_set[i] = y_dataset[i]

    x_concat = np.concatenate((x_dataset, x_rotation_set, x_colors_set), 0)
    y_concat = np.concatenate((y_dataset, y_rotation_set, y_colors_set), 0)
    shuffel = np.random.choice(y_concat.shape[0], y_concat.shape[0], replace=False)
    return x_concat[shuffel], y_concat[shuffel]

def run_final_models(mode, models_to_run, x_train, y_train, x_test, y_test, x_perturb, y_perturb, epochs=16):
    final_model = None
    l2_bn_model = None
    l2_bn_dropout_model = None
    avg_pool_model = None
    augment_spg_model = None
    augment_rc_model = None
    
    if "all" in models_to_run:
        final_model = FinalModel()
        l2_bn_model = FinalModel("l2_bn")
        l2_bn_dropout_model = FinalModel("l2_bn_dropout")
        avg_pool_model = FinalModel("avg_pool")
        augment_spg_model = FinalModel("augment_spg")
        augment_rc_model = FinalModel("augment_rc")
    else:
        if "final" in models_to_run:
            final_model = FinalModel()
        if "augment_spg" in models_to_run:
            augment_spg_model = FinalModel("augment_spg")
        if "augment_rc" in models_to_run:
            augment_rc_model = FinalModel("augment_rc")
        if "l2_bn" in models_to_run:
            l2_bn_model = FinalModel("l2_bn")
        if "l2_bn_dropout" in models_to_run:
            l2_bn_model = FinalModel("l2_bn_dropout")
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
        
        if l2_bn_dropout_model:
            l2_bn_dropout_model.train(x_train, y_train, epochs)
            l2_bn_dropout_model.evaluate(x_test, y_test)
            l2_bn_dropout_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if avg_pool_model:
            avg_pool_model.train(x_train, y_train, epochs)
            avg_pool_model.evaluate(x_test, y_test)
            avg_pool_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if augment_spg_model:
            x_train_augmented, y_train_augmented = salt_and_pepper_and_gaussian_augment(x_train, y_train)

            augment_spg_model.train(x_train_augmented, y_train_augmented, epochs)
            augment_spg_model.evaluate(x_test, y_test)
            augment_spg_model.evaluate(x_perturb, y_perturb, "Perturbed")
            del x_train_augmented, y_train_augmented
        
        if augment_rc_model:
            x_train_augmented, y_train_augmented = rotation_and_colors_augment(x_train, y_train)

            augment_rc_model.train(x_train_augmented, y_train_augmented, epochs)
            augment_rc_model.evaluate(x_test, y_test)
            augment_rc_model.evaluate(x_perturb, y_perturb, "Perturbed")
            del x_train_augmented, y_train_augmented

    elif mode == "train":
        if final_model:
            final_model.train(x_train, y_train, epochs)
        
        if l2_bn_model:
            l2_bn_model.train(x_train, y_train, epochs)
        
        if l2_bn_dropout_model:
            l2_bn_dropout_model.train(x_train, y_train, epochs)
        
        if avg_pool_model:
            avg_pool_model.train(x_train, y_train, epochs)
        
        if augment_spg_model:
            x_train_augmented, y_train_augmented = salt_and_pepper_and_gaussian_augment(x_train, y_train)

            augment_spg_model.train(x_train_augmented, y_train_augmented, epochs)
        
        if augment_rc_model:
            x_train_augmented, y_train_augmented = rotation_and_colors_augment(x_train, y_train)

            augment_rc_model.train(x_train_augmented, y_train_augmented, epochs)

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
        
        if l2_bn_dropout_model:
            l2_bn_dropout_model.evaluate(x_test, y_test)
            l2_bn_dropout_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if avg_pool_model:
            avg_pool_model.evaluate(x_test, y_test)
            avg_pool_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if augment_spg_model:
            augment_spg_model.evaluate(x_test, y_test)
            augment_spg_model.evaluate(x_perturb, y_perturb, "Perturbed")
        
        if augment_rc_model:
            augment_rc_model.evaluate(x_test, y_test)
            augment_rc_model.evaluate(x_perturb, y_perturb, "Perturbed")

    elif mode == "val_eval":
        if final_model:
            final_model.evaluate_with_validation(x_test, y_test)

if __name__ == "__main__":
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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
            if sys.argv[2].lower() == "train":
                mode = "train"
                index = 3
            elif sys.argv[2].lower() == "eval":
                mode = "eval"
                index = 3
            elif sys.argv[2].lower() == "val_train":
                mode = "val_train"
                index = 3
            elif sys.argv[2].lower() == "val_eval":
                mode = "val_eval"
                index = 3
            elif sys.argv[2].lower() == "all":
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

    #x_train, y_train = x_train[:320], y_train[0:320]

    if plot:
        augmentation_plot(x_test_RGB, x_perturb_RGB)
        exit(0)
    del x_perturb_RGB, x_train_RGB, x_test_RGB # RGB images are not needed anymore

    NUM_OF_CLASSES = 10 # CIFAR-10
    y_train = tfu.to_categorical(y_train, num_classes=NUM_OF_CLASSES) # one-hot encoding of train labels
    y_test = np.array(y_test).reshape(-1)       # test labels to 1 dimensional array
    y_perturb = np.array(y_perturb).reshape(-1) # perturb labels to 1 dimensional array

    models = Models()

    if baseline:
        models.run(mode, "baseline", x_train, y_train, x_test, y_test)
    
    if regularization:
        models.run(mode, models_to_run, x_train, y_train)
    
    if final:
        EPOCHS = 13 # number of epochs discovered as the best when training with early stopping
        run_final_models(mode, models_to_run, x_train, y_train, x_test, y_test, x_perturb, y_perturb, EPOCHS)
