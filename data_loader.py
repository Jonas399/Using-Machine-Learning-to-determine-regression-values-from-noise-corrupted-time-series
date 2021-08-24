import torch
import torch.utils.data as data
#import astropy as apy
import numpy as np
import os
from pathlib import Path
from preprocessing_Torch import preprocessDataTorch
import tensorflow as tf

class ArielDataset(data.Dataset):
    def __init__(self, path, path_params=None, start_idx=0, max_files=53900, shuffle=True, seed=None,
                 preprocessing=None, device=None):

        self.path = path
        self.preprocessing = preprocessing
        self.device = device
        self.files = sorted([element for element in os.listdir(self.path) if element.endswith('.txt')])  #glob -> siehe jupyter

        if path_params != "":
            self.path_params = path_params

        # Exception Handling
        else:
            self.path_params = None
            self.params_files = None

        # Shuffle - Funktion
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.files)
        self.files = self.files[start_idx:start_idx + max_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, file_index):
        item_path = Path(self.path) / self.files[file_index]  # "nur slash
        tensor = torch.from_numpy(np.loadtxt(item_path))
        print("hey: ", tensor)
        if self.preprocessing:
           tensor = self.preprocessing(tensor)
        if self.path_params is not None:
            item_path_params = Path(self.path_params) / self.files[file_index]
            params_tensor = torch.from_numpy(np.loadtxt(item_path_params))
        else:
            params_tensor = torch.Tensor()
        return {'tensor': tensor.to(self.device),
                'params_tensor': params_tensor.to(self.device)}
