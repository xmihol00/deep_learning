# ===========================================================================
# Authors: David Mihola, david.mihola@student.tugraz.at, 12211951
#          Massimiliano Viola, massimiliano.viola@student.tugraz.at, 12213195
# Date: 09. 11. 2022
# ===========================================================================

# The results of the experiments will be different when running on CPU, results collected for the report are from a run on a GPU. 

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 20
SUMMARY_IN_EPOCHS = [EPOCHS - 10, EPOCHS - 5, EPOCHS]
CREATE_SUMMARY = True
PLOT_GRAPHS = True
SAVE_PLOTS = True
SHOW_PLOTS = False

# to have consistent results
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

def plot_model_training(history, title="model loss", show=True, save=False): # plot of the model trainig and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    if show:
        plt.show()
    
    if save:
        plt.savefig(f"{title}.png".replace(" ", "_"))
    
    plt.close()

def plot_lr_schedules(show=True, save=False): # ploting the values of learning rate over time during training
    plt.figure(figsize=(10,6))
    steps = np.arange(len(X_train) // BATCH_SIZE * EPOCHS)
    plt.plot(steps, [0.1]*len(steps), label="fixed 0.1")
    plt.plot(steps, [0.01]*len(steps), label="fixed 0.01")
    plt.plot(steps, [0.001]*len(steps), label="fixed 0.001")
    plt.plot(steps, [0.0005]*len(steps), label="fixed 0.0005")
    plt.plot(steps, tf.keras.optimizers.schedules.ExponentialDecay(0.01, 500, 0.9)(steps),
            label="ExponentialDecay 0.01, 500, 0.9")
    plt.plot(steps, tf.keras.optimizers.schedules.CosineDecay(0.01, 4000, 0.1)(steps),
            label="CosineDecay 0.01, 4000, 0.1")
    plt.plot(steps, [tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2500, 5000], [0.1, 0.01, 0.001, 0.0005])(step) for step in steps],
            label="PiecewiseConstantDecay 0.1, 0.01, 0.001, 0.0005")
    plt.ylabel("learning rate")
    plt.xlabel("step")
    plt.title("learning rate schedules")
    plt.yscale("log")
    plt.legend()

    if show:
        plt.show()
    
    if save:
        plt.savefig(f"learning_rate_schedules.png")
    
    plt.close()
    
def create_run_model(layer_depth, neurons, number_of_epochs, X_train, Y_train): # generates and trains a model based on description in parameters, 
    model = tf.keras.models.Sequential()                                        # returns the history of the training
    model.add(tf.keras.Input(shape=(10,)))
    if type(neurons) == tuple:
        if len(neurons) != layer_depth:
            return None
        
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

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, Y_train, epochs=number_of_epochs, validation_split=0.2, batch_size=BATCH_SIZE, verbose=0)

    return history

if __name__ == "__main__":
    sc_zip_df = pd.read_csv("./data/social_capital_zip.csv")

    corr_with_sc_zip = sc_zip_df.corr()["ec_zip"]
    best_corr = corr_with_sc_zip.abs().nlargest(11).index # get the 10 most correlated variables with 'ec_zip', 'abs()' to ensure negative correlaction is included
    best_corr = list(best_corr)
    data_df = sc_zip_df[best_corr] # 10 most correlated variables with 'ec_zip' and 'ec_zip'

    data_df = data_df.dropna(subset=["ec_zip"])
    data_df = data_df.fillna(data_df.mean()) # mean imputation

    data_train_df, data_test_df = train_test_split(data_df, test_size=0.2, random_state=42) # 80/20 split of the data

    Y_train = data_train_df.pop("ec_zip")

    train_mean = data_train_df.mean()
    train_std = data_train_df.std()
    X_train = (data_train_df - train_mean) / train_std # normalization to N(0, 1)

    Y_test = data_test_df.pop("ec_zip")
    X_test = (data_test_df - train_mean) / train_std # normalization using train statistics

    # linear model baseline
    linear_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_dim=10)
    ])
    linear_model.compile(optimizer='adam', loss='mse')
    linear_history = linear_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False, validation_split=0.2)

    if PLOT_GRAPHS:
        plot_model_training(linear_history, "linear model", SHOW_PLOTS, SAVE_PLOTS)

        w, b = linear_model.weights[0].numpy(), linear_model.weights[1].numpy()
        plt.figure(figsize=(12, 12))
        plt.subplots_adjust(top=0.925, bottom=0.20, hspace=0.01, wspace=0.01)
        plt.bar(X_train.columns, w.flatten())
        plt.xticks(rotation=70)
        plt.title('linear regression coefficients')
        plt.ylabel('coefficients')

        if SHOW_PLOTS:
            plt.show()
        
        if SAVE_PLOTS:
            plt.savefig("linear_model_coefficients.png")
        
        plt.close()

    models_layers = [
        ("1 layer", 1),
        ("2 layers", 2),
        ("3 layers", 3)
    ]

    models_neurons = [
        ("all 10", 10),
        ("all 20", 20),
        ("all 30", 30),
        ("20, 10", (20, 10)),
        ("30, 20", (30, 20)),
        ("30, 20, 10", (30, 20, 10)),
        ("10, 20, 10", (10, 20, 10))
    ]

    layers_results_loss_dfs = [pd.DataFrame() for _ in SUMMARY_IN_EPOCHS]
    layers_results_validation_loss_dfs = [pd.DataFrame() for _ in SUMMARY_IN_EPOCHS]

    for layer_name, depth in models_layers:
        for neuron_name, count in models_neurons:
            history = create_run_model(depth, count, EPOCHS, X_train, Y_train)

            if history: # model was able to build and train
                train_loss, val_loss = history.history['loss'], history.history['val_loss']
                for i, epoch in enumerate(SUMMARY_IN_EPOCHS):
                    layers_results_loss_dfs[i].loc[neuron_name, layer_name] = train_loss[epoch - 1]
                    layers_results_validation_loss_dfs[i].loc[neuron_name, layer_name] = val_loss[epoch - 1]
                
                if PLOT_GRAPHS:
                    plot_model_training(history, f"{layer_name} ({neuron_name})", SHOW_PLOTS, SAVE_PLOTS)
            else: # model wasn't able to build
                for i, _ in enumerate(SUMMARY_IN_EPOCHS):
                    layers_results_loss_dfs[i].loc[neuron_name, layer_name] = None
                    layers_results_validation_loss_dfs[i].loc[neuron_name, layer_name] = None

    if CREATE_SUMMARY:
        for i, epoch in enumerate(SUMMARY_IN_EPOCHS):
            with open(f"layers_results_loss_{epoch}.html", "w") as file:
                file.write(layers_results_loss_dfs[i].style.background_gradient(cmap='Blues', axis=None).to_html())

            with open(f"layers_results_validation_loss_{epoch}.html", "w") as file:
                file.write(layers_results_validation_loss_dfs[i].style.background_gradient(cmap='Blues', axis=None).to_html())

    names_lr_schedules = [
        ("fixed 0.1", 0.1),
        ("fixed 0.01", 0.01),
        ("fixed 0.001", 0.001),
        ("fixed 0.0005", 0.0005),
        ("ExponentialDecay 0.01, 500, 0.9", tf.keras.optimizers.schedules.ExponentialDecay(0.01, 500, 0.9)),
        ("CosineDecay 0.01, 4000, 0.1", tf.keras.optimizers.schedules.CosineDecay(0.01, 4000, 0.1)),
        ("PiecewiseConstantDecay 0.1, 0.01, 0.001, 0.0005", tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2500, 5000], [0.1, 0.01, 0.001, 0.0005]))
    ]

    optimizers_momentum = [
        ("SGD", 0.0),
        ("SGD momentum 0.9", 0.9),
        ("SGD momentum 0.75", 0.75),
        ("ADAM", None)
    ]

    if PLOT_GRAPHS:
        plot_lr_schedules(SHOW_PLOTS, SAVE_PLOTS)

    optimizer_results_loss_dfs = [pd.DataFrame() for _ in SUMMARY_IN_EPOCHS]
    optimizer_results_validation_loss_dfs = [pd.DataFrame() for _ in SUMMARY_IN_EPOCHS]

    for optimizer_name, momentum in optimizers_momentum:
        for lr_name, lr_schedule in names_lr_schedules:
            chosen_model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(20, input_dim=10, activation='relu'),
                tf.keras.layers.Dense(1),
            ])

            if "SGD" in optimizer_name:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)
            elif "ADAM" in optimizer_name:
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            chosen_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
            history = chosen_model.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, batch_size=BATCH_SIZE, verbose=0)

            for i, epoch in enumerate(SUMMARY_IN_EPOCHS):
                optimizer_results_loss_dfs[i].loc[lr_name, optimizer_name] = history.history['loss'][epoch - 1]
                optimizer_results_validation_loss_dfs[i].loc[lr_name, optimizer_name] = history.history['val_loss'][epoch - 1]

    if CREATE_SUMMARY:
        for i, epoch in enumerate(SUMMARY_IN_EPOCHS):
            with open(f"optimizer_results_loss_{epoch}.html", "w") as file:
                file.write(optimizer_results_loss_dfs[i].style.background_gradient(cmap='Blues', axis=None).to_html())

            with open(f"optimizer_results_validation_loss_{epoch}.html", "w") as file:
                file.write(optimizer_results_validation_loss_dfs[i].style.background_gradient(cmap='Blues', axis=None).to_html())

    final_model = tf.keras.models.Sequential([ # model with the best performance
        tf.keras.layers.Dense(20, input_dim=10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 500, 0.9) # learning rate schedule with the best performance
    final_model.compile(optimizer=tf.keras.optimizers.Adam(scheduler), loss='mse')
    final_history = final_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False)

    if PLOT_GRAPHS:
        plt.figure(figsize=(10, 5))
        plt.plot(final_history.history['loss'])
        plt.title('chosen model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')

        if SHOW_PLOTS:
            plt.show()
    
        if SAVE_PLOTS:
            plt.savefig(f"final_model_loss.png")

        plt.close()

    Y_pred = final_model.predict(X_test).flatten() # predicitng the values for test set
    print(f"Test MSE: {np.mean(np.square(Y_pred - Y_test))}") # see the error

    if PLOT_GRAPHS:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.scatter(Y_test, Y_pred)
        ax1.set_title('Ground truth vs predicted')
        ax1.set_xlabel('Ground truth')
        ax1.set_ylabel('Predicted')
        ax2.hist(Y_pred-Y_test, bins=50)
        ax2.set_title('Difference between ground truth and predicted')
        ax2.set_ylabel('Number of samples')
        ax2.set_xlabel('Difference')

        if SHOW_PLOTS:
            plt.show()

        if SAVE_PLOTS:
            plt.savefig(f"ground_truth_predicted_comparison.png")

        plt.close()
