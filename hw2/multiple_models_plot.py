import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

BATCH_SIZE = 32
EPOCHS = 20

# to have consistent results
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

def plot_model_training(history, title="model loss"):
    plt.figure(figsize=(12, 6))
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.title(title)
    plt.ylabel('logarithmic loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()


def create_run_model(layer_depth, neurons, X_train, Y_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(10,)))
    if type(neurons) == tuple:
        if len(neurons) != layer_depth:
            return (None, None)
        
        for neuron_count in neurons:
            model.add(tf.keras.layers.Dense(neuron_count, activation='relu'))  
            model.add(tf.keras.layers.Dropout(0.4))
        model.pop()
    else:
        for _ in range(layer_depth):
            model.add(tf.keras.layers.Dense(neurons, activation='relu'))  
            model.add(tf.keras.layers.Dropout(0.4))
        model.pop()
    
    model.add(tf.keras.layers.Dense(1))
    print(model.summary())

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, batch_size=BATCH_SIZE, verbose=0)

    #plot_model_training(history)

    return history.history['loss'], history.history['val_loss']

if __name__ == "__main__":
    sc_zip_df = pd.read_csv("./data/social_capital_zip.csv")

    corr_with_sc_zip = sc_zip_df.corr()["ec_zip"]
    best_corr = corr_with_sc_zip.abs().nlargest(11).index # get the 10 most correlated variables with 'ec_zip', 'abs()' to ensure negative correlaction is included
    best_corr = list(best_corr)
    data_df = sc_zip_df[best_corr]

    data_df = data_df.dropna(subset=["ec_zip"])
    data_df = data_df.fillna(data_df.mean())

    data_train_df, data_test_df = train_test_split(data_df, test_size=0.2, random_state=42)

    Y_train = data_train_df.pop("ec_zip")

    train_mean = data_train_df.mean()
    train_std = data_train_df.std()
    X_train = (data_train_df - train_mean) / train_std # normalization to N(0, 1)

    Y_test = data_test_df.pop("ec_zip")
    X_test = (data_test_df - train_mean) / train_std # normalization using train statistics

    figure, axis = plt.subplots(2, 2)
    figure.set_size_inches(25, 22)
    for layers, neurons, name, i, j in [(1, 20, "model with 1 hidden layer of 20 units", 0, 0), (2, 30, "model with 2 hidden layers of 30 units", 0, 1), 
                                        (2, (30, 20), "model with 2 hidden layers first of 30 units, second of 20 units", 1, 1), (3, 10, "model with 3 hidden layers of 10 units", 1, 0)]:
        train_loss, valid_loss = create_run_model(layers, neurons, X_train, Y_train)
        ax = axis[i, j]
        ax.set_title(name)
        ax.set_ylabel('logarithmic loss')
        ax.set_xlabel('epoch')
        ax.plot(np.log(train_loss))
        ax.plot(np.log(valid_loss))
        ax.legend(['training', 'validation'], loc='upper right')

    plt.show()
