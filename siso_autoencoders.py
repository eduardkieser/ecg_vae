from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
# from matplotlib import use
from torchsummary import summary
# use('MacOSX')

class VAE_siso_dense(nn.Module):
    def __init__(
            self,
            size_oi = 280,
            size_g=100,
            size_z=5,
            n_grouped_input_layers=3,
            n_grouped_output_layers=3,
            name='SISO_VAE_DENSE'
    ):
        super(VAE_siso_dense, self).__init__()
        self.size_oi = size_oi
        self.size_g = size_g
        self.size_z = size_z
        self.n_grouped_input_layers = n_grouped_input_layers
        self.n_grouped_output_layers = n_grouped_output_layers
        self.name = name
        self.base_architecture = 'SISO_VAE_DENSE'
        self.build_graph_elements()


    def build_graph_elements(self):
        self.dense_p_11 = nn.Linear(self.size_g, self.size_z)
        self.dense_p_12 = nn.Linear(self.size_g, self.size_z)

        self.input_layers = nn.ModuleList([nn.Linear(self.size_oi, self.size_g)])
        for i in range(self.n_grouped_input_layers):
            self.input_layers.append(
                nn.Linear(self.size_g, self.size_g)
            )

        self.output_layers = nn.ModuleList([nn.Linear(self.size_z, self.size_g)])
        for i in range(self.n_grouped_output_layers):
            self.output_layers.append(
                nn.Linear(self.size_g, self.size_g)
            )
        self.output_layers.append(nn.Linear(self.size_g, self.size_oi))


    def encode(self, x):
        for fi in self.input_layers:
            x = F.relu(fi(x))
        return self.dense_p_11(x), self.dense_p_12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = z
        for fo in self.output_layers[:-1]:
            x = F.relu(fo(x))
        return self.output_layers[-1](x)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 280))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    ################################


if __name__=='__main__':
    model = VAE_siso_dense()
    summary(model, input_size=(128,280))