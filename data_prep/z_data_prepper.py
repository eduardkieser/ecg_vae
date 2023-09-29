
import sys

sys.path.append('../')
from ecg_loader_config import ECGDataConfig, DataSelectionMode, ModelOperationMode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import torch
from autoencoders import VAE_dense3

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_gpu else "cpu")

class ZDataPrepper():

    def __init__(self):
        self.rename_map = {f'{num}':f'wl_{num}' for num in range(300)}
        self.wavelet_columns = list(self.rename_map.values())
        self.rename_map_zs =  {num:f'z_{num}' for num in range(5)}
        self.z_columns = list(self.rename_map_zs.values())
        self.load_model()


    def load_model(self):
        state_dict_path = '../models/VAE_dense3_V1_center/checkpoints/state_dict_latest.tar'
        checkpoint = torch.load(state_dict_path)
        model = VAE_dense3()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(device).double()
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']

    def load_data(self):
        data_path = '../data/training_data/V1_center.h5'
        self.data_raw = pd.read_hdf(data_path, key='df')
        self.data_raw = self.data_raw.rename(axis=1, mapper=self.rename_map)
        # self.data_raw = self.data_raw.sample(20000)


    def append_zs(self):
        z_df_list = []
        for ix, row in self.data_raw.iterrows():
            wavelet = row[self.wavelet_columns]
            wavelet = wavelet.iloc[10:290]
            wavelet = torch.DoubleTensor(np.array(wavelet).astype(float))
            mu, logvar = self.model.encode(wavelet)
            zs = self.model.reparameterize(mu, logvar).detach().numpy().astype(float)
            z_df_list.append(zs)
            if ix %1000==0:
                print(f'ix={ix}')
        zs_df = pd.DataFrame(z_df_list)\
            .reset_index(drop=True)\
            .rename(axis=1,mapper=self.rename_map_zs)
        self.zs_df = pd.concat([
            self.data_raw.reset_index(drop=True),
            zs_df
        ],axis=1)


    def save_zs_df(self):
        zs_df_path = '../data/training_data/V1_center_with_zs.h5'
        self.zs_df.to_hdf(zs_df_path,key='df')


    def load_zs_df(self):
        zs_df_path = '../data/training_data/V1_center_with_zs.h5'
        self.zs_df = pd.read_hdf(zs_df_path,key='df')
        j=2


    def prep_z_df(self):
        self.load_data()
        self.append_zs()
        self.save_zs_df()






if __name__=='__main__':
    prepper = ZDataPrepper()

