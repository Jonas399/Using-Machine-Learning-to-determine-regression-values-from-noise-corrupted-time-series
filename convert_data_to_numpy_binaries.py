# -*- coding: utf-8 -*-


# aktuelle pandas version notwendig für to_numpy()
# lokale anaconda env:  env_test 

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import glob  # for data import

import time


from pathlib import Path
path_home = str(Path.home())

path_out = path_home + f"/planets_ECML2021/conv/"

#path_out = path_home + f"/planets_ECML2021/conv/tmp/"


# in

path_data = 'C:/Users/anth/Dropbox/myresearch/_datasets/planets_ECML2021/'

path = path_data + "_small_data/" # use your path

path = path_data + "training_set/noisy_train/" # use your path
all_training_files = glob.glob(path + "/*.txt")

path_params = path_data + "params_train/" # use your path
params_files = glob.glob(path_params + "/*.txt")

path = path_data + "test_set/noisy_test/"
test_files = glob.glob(path + "/*.txt")


#import data  (for training set)
files_to_import = 200*100 # len(all_training_files)  # all, or any number...




#### training data 
print(f"Importing training data (raw files): {files_to_import} files")
training_list = []

for idxFile in range(0, files_to_import):
  df = pd.read_csv(all_training_files[idxFile], header=None, sep="\t", skiprows=6)
  if idxFile % 100 == 0:
      print("")
  print(".", end='')
  training_list.append(df)

train_data_in = pd.concat(training_list, axis=0, ignore_index=True)
print("")
print(f"Import finished: {files_to_import} files imported. Data shape: {train_data_in.shape}")


#filename_out = path_out + f"train_{int(files_to_import/100)}_planets.csv"
# data frame to CSV
#train_data_in.to_csv(filename_out, index=False, header=False)
#loaded_data_df = pd.read_csv(filename_out, header=None)

# save as numpy 
filename_out = path_out + f"train_{int(files_to_import/100)}_planets.npy"
with open(filename_out, 'wb') as f:
    np.save(f, train_data_in.to_numpy())

loaded_data_npy = np.load(filename_out)


######### training targets
print(f"Importing labels (params_train): {files_to_import} files")
params_list = []

for idxFile in range(0, files_to_import):
  df = pd.read_csv(params_files[idxFile], header=None, sep="\t", skiprows=2)
  print(".", end='')
  params_list.append(df)


train_target_in = pd.concat(params_list, axis=0, ignore_index=True)
print("")
print(f"Import finished: {files_to_import} files imported. Data shape: {train_target_in.shape}")


# save as numpy 
filename_out = path_out + f"target_{int(files_to_import/100)}_planets.npy"
with open(filename_out, 'wb') as f:
    np.save(f, train_target_in.to_numpy())

loaded_target_npy = np.load(filename_out)


# data frame to CSV
#filename_out = path_out + f"target_{int(files_to_import/100)}_planets.csv"
#train_target_in.to_csv(filename_out, index=False, header=False)
#loaded_targets_df = pd.read_csv(filename_out, header=None)




### challenge test set 
    
# n_test_files = len(test_files)  # all, or any number...    
    
# print(f"Importing challenge test set: {n_test_files} files")
# test_list = []


# for idxFile in range(0, n_test_files):
#   df = pd.read_csv(test_files[idxFile], header=None, sep="\t", skiprows=6)
#   if idxFile % 100 == 0:
#       print("")
#   print(".", end='')
#   test_list.append(df)

# test_in = pd.concat(test_list, axis=0, ignore_index=True)
# print("")
# print(f"Import finished: {n_test_files} files imported. Data shape: {test_in.shape}")


# # save as numpy 
# filename_out = path_out + f"testset_challenge.npy"
# with open(filename_out, 'wb') as f:
#     np.save(f, test_in.to_numpy())

# loaded_test_npy = np.load(filename_out)



###### final test set


# n_files_final_test = 200 * 100  # N planets * 100

# print(f"Importing final test data (raw files): {n_files_final_test} files")


# final_test_list = []

# for idxFile in range(len(all_training_files) - n_files_final_test, len(all_training_files)):
#   df = pd.read_csv(all_training_files[idxFile], header=None, sep="\t", skiprows=6)
#   if idxFile % 100 == 0:
#       print("")
#   print(".", end='')
#   final_test_list.append(df)

# final_test_in = pd.concat(final_test_list, axis=0, ignore_index=True)
# print("")
# print(f"Import finished: {n_files_final_test} files imported. Data shape: {final_test_in.shape}")


# # save as numpy 
# filename_out = path_out + f"test_final_{int(n_files_final_test/100)}_planets.npy"
# with open(filename_out, 'wb') as f:
#     np.save(f, final_test_in.to_numpy())

# #loaded_final_test_npy = np.load(filename_out)


# params_final_test_list = []

# for idxFile in range(len(all_training_files) - n_files_final_test, len(all_training_files)):
#   df = pd.read_csv(params_files[idxFile], header=None, sep="\t", skiprows=2)
#   print(".", end='')
#   params_final_test_list .append(df)


# final_test_target_in = pd.concat(params_final_test_list , axis=0, ignore_index=True)
# print("")
# print(f"Import finished: {n_files_final_test} files imported. Data shape: {final_test_target_in.shape}")


# # save as numpy 
# filename_out = path_out + f"test_target_final_{int(n_files_final_test/100)}_planets.npy"
# with open(filename_out, 'wb') as f:
#     np.save(f, final_test_target_in.to_numpy())

#loaded_final_test_target_npy = np.load(filename_out)



