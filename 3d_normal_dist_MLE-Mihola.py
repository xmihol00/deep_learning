import numpy as np
import pandas as pd

# Change to 'True', to double check the result.
WITH_CHECKS = False

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
