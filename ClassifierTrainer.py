from torch import nn, optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoencoders import VAE_frank1
import torch
use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_gpu else "cpu")
import os


class ClassifierTrainer(nn.Module):

    def __init__(self):
        super().__init__()

    def load_model(self):
        state_dict_path = 'models/VAE_dense3_V1_center/checkpoints/state_dict_latest_.tar'
        checkpoint = torch.load(state_dict_path)
        model = VAE_frank1()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(device).double()
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']


    def setup_model_dir(self):
        self.model_folder_path = f'models/{self.model.name}_' \
            f'{mixed_data_config_training.lead_name.value}_' \
            f'{mixed_data_config_training.window_alignment.value}'

        # create forder if it doesn't exist.
        os.makedirs(f'{self.model_folder_path}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.model_folder_path}/figures', exist_ok=True)


    def initialize_model_and_optimizer(self):
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.double()
            self.lr = 1e-3
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.setup_model_dir()
        if self.use_cached:
            self.load_state_dict_if_available()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr/10)