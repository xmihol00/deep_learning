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

def train_models(models, models_to_train, x_train, y_train):
    # ====================================================== baseline ======================================================
    if "baseline" in models_to_train or "all" in models_to_train:
        print("\n\nbaseline models training:")
        models.FC_SP_16_256.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256.save_weights("./models/FC_SP_16_256/FC_SP_16_256")
        del models.FC_SP_16_256

        models.FC_MP_16_256.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256.save_weights("./models/FC_MP_16_256/FC_MP_16_256")
        del models.FC_MP_16_256

        models.FC_MP_32_512.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512.save_weights("./models/FC_MP_32_512/FC_MP_32_512")
        del models.FC_MP_32_512

        models.VGG_2B_32_64.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64")
        del models.VGG_2B_32_64

        models.VGG_3B_16_64.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64")
        del models.VGG_3B_16_64

        models.VGG_3B_32_128.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128")
        del models.VGG_3B_32_128

    # ================================================= batch norm, dropout 0.3 =================================================
    if "dropout_03" in models_to_train or "all" in models_to_train:
        print("\n\nbatch norm, dropout_03 models training:")
        models.FC_SP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_batch_norm.save_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm")
        del models.FC_SP_16_256_batch_norm

        models.FC_MP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_batch_norm.save_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm")
        del models.FC_MP_16_256_batch_norm

        models.FC_MP_32_512_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_batch_norm.save_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm")
        del models.FC_MP_32_512_batch_norm

        models.VGG_2B_32_64_dropout_03.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_dropout_03.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_dropout_03.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_03")
        del models.VGG_2B_32_64_dropout_03

        models.VGG_3B_16_64_dropout_03.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_dropout_03.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_dropout_03.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_03")
        del models.VGG_3B_16_64_dropout_03

        models.VGG_3B_32_128_dropout_03.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_dropout_03.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_dropout_03.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_03")
        del models.VGG_3B_32_128_dropout_03

    # ================================================= batch norm, dropout 0.4 =================================================
    if "dropout_04" in models_to_train or "all" in models_to_train:
        print("\n\nbatch norm, dropout_04 models training:")
        models.FC_SP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_batch_norm.save_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm")
        del models.FC_SP_16_256_batch_norm

        models.FC_MP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_batch_norm.save_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm")
        del models.FC_MP_16_256_batch_norm

        models.FC_MP_32_512_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_batch_norm.save_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm")
        del models.FC_MP_32_512_batch_norm

        models.VGG_2B_32_64_dropout_04.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_dropout_04.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_dropout_04.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_04")
        del models.VGG_2B_32_64_dropout_04

        models.VGG_3B_16_64_dropout_04.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_dropout_04.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_dropout_04.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_04")
        del models.VGG_3B_16_64_dropout_04

        models.VGG_3B_32_128_dropout_04.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_dropout_04.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_dropout_04.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_04")
        del models.VGG_3B_32_128_dropout_04

    # ================================================= batch norm, dropout 0.5 =================================================
    if "dropout_05" in models_to_train or "all" in models_to_train:
        print("\n\ndropout_05 models training:")
        models.FC_SP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_batch_norm.save_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm")
        del models.FC_SP_16_256_batch_norm


        models.FC_MP_16_256_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_batch_norm.save_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm")
        del models.FC_MP_16_256_batch_norm

        models.FC_MP_32_512_batch_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_batch_norm.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_batch_norm.save_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm")
        del models.FC_MP_32_512_batch_norm

        models.VGG_2B_32_64_dropout_05.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_dropout_05.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_dropout_05.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_05")
        del models.VGG_2B_32_64_dropout_05

        models.VGG_3B_16_64_dropout_05.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_dropout_05.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_dropout_05.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_05")
        del models.VGG_3B_16_64_dropout_05

    models.VGG_3B_32_128_dropout_05.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    models.VGG_3B_32_128_dropout_05.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                              callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
    models.VGG_3B_32_128_dropout_05.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_05")
    del models.VGG_3B_32_128_dropout_05

    # ====================================================== l1l2 0.0001 ======================================================
    if "l1l2_00001" in models_to_train or "all" in models_to_train:
        print("\n\nl1l2_00001 models training:")
        models.FC_SP_16_256_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l1l2_00001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_00001")
        del models.FC_SP_16_256_l1l2_00001

        models.FC_MP_16_256_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l1l2_00001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_00001")
        del models.FC_MP_16_256_l1l2_00001

        models.FC_MP_32_512_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l1l2_00001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_00001")
        del models.FC_MP_32_512_l1l2_00001

        models.VGG_2B_32_64_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l1l2_00001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_00001")
        del models.VGG_2B_32_64_l1l2_00001

        models.VGG_3B_16_64_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l1l2_00001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_00001")
        del models.VGG_3B_16_64_l1l2_00001

        models.VGG_3B_32_128_l1l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l1l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l1l2_00001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_00001")
        del models.VGG_3B_32_128_l1l2_00001

    # ====================================================== l1l2 0.001 ======================================================
    if "l1l2_0001" in models_to_train or "all" in models_to_train:
        print("\n\nl1l2_0001 models training:")
        models.FC_SP_16_256_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l1l2_0001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_0001")
        del models.FC_SP_16_256_l1l2_0001

        models.FC_MP_16_256_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l1l2_0001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_0001")
        del models.FC_MP_16_256_l1l2_0001

        models.FC_MP_32_512_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l1l2_0001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_0001")
        del models.FC_MP_32_512_l1l2_0001

        models.VGG_2B_32_64_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l1l2_0001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_0001")
        del models.VGG_2B_32_64_l1l2_0001

        models.VGG_3B_16_64_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l1l2_0001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_0001")
        del models.VGG_3B_16_64_l1l2_0001

        models.VGG_3B_32_128_l1l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l1l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l1l2_0001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_0001")
        del models.VGG_3B_32_128_l1l2_0001

    # ====================================================== l1l2 0.01 ======================================================
    if "l1l2_001" in models_to_train or "all" in models_to_train:
        print("\n\nl1l2_001 models training:")
        models.FC_SP_16_256_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l1l2_001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_001")
        del models.FC_SP_16_256_l1l2_001

        models.FC_MP_16_256_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l1l2_001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_001")
        del models.FC_MP_16_256_l1l2_001

        models.FC_MP_32_512_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l1l2_001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_001")
        del models.FC_MP_32_512_l1l2_001

        models.VGG_2B_32_64_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l1l2_001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_001")
        del models.VGG_2B_32_64_l1l2_001

        models.VGG_3B_16_64_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l1l2_001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_001")
        del models.VGG_3B_16_64_l1l2_001

        models.VGG_3B_32_128_l1l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l1l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l1l2_001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_001")
        del models.VGG_3B_32_128_l1l2_001

    # ====================================================== l1 0.0001 ======================================================
    if "l1_00001" in models_to_train or "all" in models_to_train:
        print("\n\nl1_00001 models training:")
        models.FC_SP_16_256_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l1_00001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_00001")
        del models.FC_SP_16_256_l1_00001

        models.FC_MP_16_256_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l1_00001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_00001")
        del models.FC_MP_16_256_l1_00001

        models.FC_MP_32_512_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l1_00001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_00001")
        del models.FC_MP_32_512_l1_00001

        models.VGG_2B_32_64_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l1_00001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_00001")
        del models.VGG_2B_32_64_l1_00001

        models.VGG_3B_16_64_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l1_00001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_00001")
        del models.VGG_3B_16_64_l1_00001

        models.VGG_3B_32_128_l1_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l1_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l1_00001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_00001")
        del models.VGG_3B_32_128_l1_00001

    # ====================================================== l1 0.001 ======================================================
    if "l1_0001" in models_to_train or "all" in models_to_train:
        print("\n\nl1_0001 models training:")
        models.FC_SP_16_256_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l1_0001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_0001")
        del models.FC_SP_16_256_l1_0001

        models.FC_MP_16_256_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l1_0001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_0001")
        del models.FC_MP_16_256_l1_0001

        models.FC_MP_32_512_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l1_0001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_0001")
        del models.FC_MP_32_512_l1_0001

        models.VGG_2B_32_64_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l1_0001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_0001")
        del models.VGG_2B_32_64_l1_0001

        models.VGG_3B_16_64_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l1_0001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_0001")
        del models.VGG_3B_16_64_l1_0001

        models.VGG_3B_32_128_l1_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l1_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l1_0001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_0001")
        del models.VGG_3B_32_128_l1_0001

    # ====================================================== l1 0.01 ======================================================
    if "l1_001" in models_to_train or "all" in models_to_train:
        print("\n\nl1_001 models training:")
        models.FC_SP_16_256_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l1_001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_001")
        del models.FC_SP_16_256_l1_001

        models.FC_MP_16_256_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l1_001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_001")
        del models.FC_MP_16_256_l1_001

        models.FC_MP_32_512_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l1_001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_001")
        del models.FC_MP_32_512_l1_001

        models.VGG_2B_32_64_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l1_001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_001")
        del models.VGG_2B_32_64_l1_001

        models.VGG_3B_16_64_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l1_001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_001")
        del models.VGG_3B_16_64_l1_001

        models.VGG_3B_32_128_l1_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l1_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l1_001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_001")
        del models.VGG_3B_32_128_l1_001

    # ====================================================== l2 0.0001 ======================================================
    if "l2_00001" in models_to_train or "all" in models_to_train:
        print("\n\nl2_00001 models training:")
        models.FC_SP_16_256_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l2_00001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_00001")
        del models.FC_SP_16_256_l2_00001

        models.FC_MP_16_256_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l2_00001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_00001")
        del models.FC_MP_16_256_l2_00001

        models.FC_MP_32_512_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l2_00001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_00001")
        del models.FC_MP_32_512_l2_00001

        models.VGG_2B_32_64_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l2_00001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_00001")
        del models.VGG_2B_32_64_l2_00001

        models.VGG_3B_16_64_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l2_00001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_00001")
        del models.VGG_3B_16_64_l2_00001

        models.VGG_3B_32_128_l2_00001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l2_00001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l2_00001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_00001")
        del models.VGG_3B_32_128_l2_00001

    # ====================================================== l2 0.001 ======================================================
    if "l2_0001" in models_to_train or "all" in models_to_train:
        print("\n\nl2_0001 models training:")
        models.FC_SP_16_256_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l2_0001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_0001")
        del models.FC_SP_16_256_l2_0001

        models.FC_MP_16_256_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l2_0001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_0001")
        del models.FC_MP_16_256_l2_0001

        models.FC_MP_32_512_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l2_0001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_0001")
        del models.FC_MP_32_512_l2_0001

        models.VGG_2B_32_64_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l2_0001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_0001")
        del models.VGG_2B_32_64_l2_0001

        models.VGG_3B_16_64_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l2_0001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_0001")
        del models.VGG_3B_16_64_l2_0001

        models.VGG_3B_32_128_l2_0001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l2_0001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l2_0001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_0001")
        del models.VGG_3B_32_128_l2_0001

    # ====================================================== l2 0.01 ======================================================
    if "l2_001" in models_to_train or "all" in models_to_train:
        print("\n\nl2_001 models training:")
        models.FC_SP_16_256_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_SP_16_256_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                      callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_SP_16_256_l2_001.save_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_001")
        del models.FC_SP_16_256_l2_001

        models.FC_MP_16_256_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_16_256_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_16_256_l2_001.save_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_001")
        del models.FC_MP_16_256_l2_001

        models.FC_MP_32_512_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.FC_MP_32_512_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                     callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.FC_MP_32_512_l2_001.save_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_001")
        del models.FC_MP_32_512_l2_001

        models.VGG_2B_32_64_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_2B_32_64_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_2B_32_64_l2_001.save_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_001")
        del models.VGG_2B_32_64_l2_001

        models.VGG_3B_16_64_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_16_64_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                  callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_16_64_l2_001.save_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_001")
        del models.VGG_3B_16_64_l2_001

        models.VGG_3B_32_128_l2_001.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        models.VGG_3B_32_128_l2_001.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,
                                        callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)])
        models.VGG_3B_32_128_l2_001.save_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_001")
        del models.VGG_3B_32_128_l2_001
