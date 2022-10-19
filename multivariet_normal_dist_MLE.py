from re import A
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sc_zip_df = pd.read_csv("data/social_capital_zip.csv")

ec_zip = sc_zip_df[["ec_zip"]].values.reshape(-1)
ec_grp_mem_zip = sc_zip_df[["ec_grp_mem_zip"]].values.reshape(-1)
exposure_grp_mem_zip = sc_zip_df[["exposure_grp_mem_zip"]].values.reshape(-1)

# remove NaN values
ec_zip = ec_zip[~np.isnan(ec_zip)]
ec_grp_mem_zip = ec_grp_mem_zip[~np.isnan(ec_grp_mem_zip)]
exposure_grp_mem_zip = exposure_grp_mem_zip[~np.isnan(exposure_grp_mem_zip)]

min_samples = np.min([ec_zip.shape[0], ec_grp_mem_zip.shape[0], exposure_grp_mem_zip.shape[0]])
print(min_samples)