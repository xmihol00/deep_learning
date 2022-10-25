# ========================================================
# Author: David Mihola (david.mihola@student.tugraz.at)
# Matrikelnummer: 12211951
# Date: 09. 11. 2022
# ========================================================

from math import fabs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change to 'True', to double check the result.
WITH_CHECKS = False
# Change to 'True', to plot the values and estimated 3d mean.
WITH_PLOT = False

# Load chosen data set -- move the file to the used location, or change the code.
sc_zip_df = pd.read_csv("data/social_capital_zip.csv")

# Load chosen columns.
data = sc_zip_df[["ec_zip", "ec_grp_mem_zip", "exposure_grp_mem_zip"]]

# Remove rows with NaN values.
data = data.dropna()

# Get the numpy matrix.
data = data.values

if WITH_CHECKS:
    # Check that there are no NaN values
    print(f"Number of NaN values: {np.isnan(data.shape).sum()}")

# Get the number of samples.
N = data.shape[0]

# Implement the derived maximum-likelihood estimate: mu_MLE = sum(x^n)/N, where x^n is the nth 3d vector and N is the number of samples.
mu_MLE = np.sum(data, axis=0) / N
print(f"The maximum-likelihood estimate of the mean vector is: {mu_MLE}.")

if WITH_PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # randomly select 150 samples
    np.random.seed(42)
    rand_150_data = data[np.random.randint(N, size=150)] 
    # plot the samples
    ax.scatter(rand_150_data[:, 0], rand_150_data[:, 1], rand_150_data[:, 2])
    # plot the mean
    ax.scatter(mu_MLE[0], mu_MLE[1], mu_MLE[2], color="orange", s=250, alpha=1)
    ax.set_xlabel('ec_zip')
    ax.set_ylabel('ec_grp_mem_zip')
    ax.set_zlabel('exposure_grp_mem_zip')
    ax.set_title("150 randomly selected samples (blue) and the estimated mean (orange).")
    plt.show()

if WITH_CHECKS:
    # The MLE mean estimate claculated by a numpy function.
    print(f"The MLE of the mean vector using numpy is: {np.mean(data, axis=0)}.")

    # The MLE estimate of the marginal means, differs slightly as there are less samples dropped due to NaN values
    ec_zip = sc_zip_df[["ec_zip"]].dropna().values.reshape(-1)
    print(f"The marginal MLE of the mean_1 is: {np.mean(ec_zip)}.")
    ec_grp_mem_zip = sc_zip_df[["ec_grp_mem_zip"]].dropna().values.reshape(-1)
    print(f"The marginal MLE of the mean_2 is: {np.mean(ec_grp_mem_zip)}.")
    exposure_grp_mem_zip = sc_zip_df[["exposure_grp_mem_zip"]].dropna().values.reshape(-1)
    print(f"The marginal MLE of the mean_3 is: {np.mean(exposure_grp_mem_zip)}.")
