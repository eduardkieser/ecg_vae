from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from matplotlib import use
from torchsummary import summary
from typing import List, Tuple
# use('MacOSX')


class VAE_mimo_dense(nn.Module):
    def __init__(
            self,
            size_oi=280,
            size_g=70,
            size_i=70,
            size_z=5,
            n_individual_input_layers=1,
            n_grouped_input_layers=1,
            n_individual_output_layers=1,
            n_grouped_output_layers=1,
            name='MIMO_VAE_DENSE',
    ):
        super(VAE_mimo_dense, self).__init__()

        self.size_oi = size_oi  # input/output / wavelet size
        self.size_g = size_g  # intermediate grouped size
        self.size_i = size_i  # intermediate individual size
        self.size_z = size_z  # latent variable space size

        self.n_input_channels = 2
        self.n_individual_input_layers = n_individual_input_layers
        self.n_grouped_input_layers = n_grouped_input_layers
        self.n_individual_output_layers = n_individual_output_layers
        self.n_grouped_output_layers = n_grouped_output_layers
        self.n_output_channels = 2

        self.name = f'{name}_{n_individual_input_layers}x{size_i}_{n_grouped_input_layers}x{size_g}_{size_z}'
        self.base_architecture = 'MIMO_VAE_DENSE'
        self.output_scalar = torch.tensor([6])

        self.create_graph_elements()

    def create_graph_elements(self):
        size_io, size_z, size_g, size_i = self.size_oi, self.size_z, self.size_g, self.size_i

        # create individual input layers
        self.individual_input_layers_0 =nn.ModuleList([nn.Linear(size_io,size_i)])
        self.individual_input_layers_1 = nn.ModuleList([nn.Linear(size_io, size_i)])
        for i in range(self.n_individual_input_layers-1):
            self.individual_input_layers_0.append(nn.Linear(size_i,size_i))
            self.individual_input_layers_1.append(nn.Linear(size_i, size_i))
        # create grouped input layers
        self.grouped_input_layers = nn.ModuleList([nn.Linear(2*size_i,size_g)])
        for i in range(self.n_grouped_input_layers-1):
            self.grouped_input_layers.append(
                nn.Linear(size_g,size_g)
            )
        # create grouped output layers
        self.grouped_output_layers = nn.ModuleList([nn.Linear(size_z, size_g)])
        for i in range(self.n_grouped_output_layers-1):
            self.grouped_output_layers.append(nn.Linear(size_g,size_g))

        # create individual output layers
        if self.n_individual_output_layers>=2:
            self.individual_output_layers_0 = nn.ModuleList([nn.Linear(size_g,size_i)])
            self.individual_output_layers_1 = nn.ModuleList([nn.Linear(size_g, size_i)])
            for i in range(self.n_individual_output_layers-2):
                self.individual_output_layers_0.append(nn.Linear(size_i, size_i))
                self.individual_output_layers_1.append(nn.Linear(size_i, size_i))
            self.individual_output_layers_0.append(nn.Linear(size_i, size_io))
            self.individual_output_layers_1.append(nn.Linear(size_i, size_io))
        else:
            self.individual_output_layers_0 = nn.ModuleList([nn.Linear(size_g, size_io)])
            self.individual_output_layers_1 = nn.ModuleList([nn.Linear(size_g, size_io)])
        
        self.dense_p_11, self.dense_p_12 = nn.Linear(size_g,size_z), nn.Linear(size_g,size_z)


    def encode(self, x1, x2):
        for (dense1, dense2) in zip(self.individual_input_layers_0, self.individual_input_layers_1):
            x1 = F.relu(dense1(x1))
            x2 = F.relu(dense2(x2))
        x = torch.cat([x1,x2],dim=1)
        for lin in self.grouped_input_layers:
            x = F.relu(lin(x))

        return self.dense_p_11(x), self.dense_p_12(x)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x=z
        for dense in self.grouped_output_layers:
            x = F.relu(dense(x))
        x1=x;x2=x
        for (dense1, dense2) in zip(self.individual_output_layers_0[:-1],self.individual_output_layers_1[:-1]):
            x1 = F.relu(dense1(x1))
            x2 = F.relu(dense2(x2))
        dense1 = self.individual_output_layers_0[-1]
        dense2 = self.individual_output_layers_1[-1]

        # # having tanh as activation limits output to -1&1
        # # input rages from ± -6 to 6 therefore multiply...
        # sc = self.output_scalar
        # return torch.tanh(dense1(x1)) * sc, torch.tanh(dense2(x2)) * sc

        return dense1(x1), dense2(x2)

    def forward(self, x1,x2):
        mu, logvar = self.encode(x1.view(-1, 280),x2.view(-1, 280))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


if __name__=='__main__':
    model = VAE_mimo_dense(size_g=50)
    summary(model, input_size=[(128,280),(128,280)])
    # print(model)