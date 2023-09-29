from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from matplotlib import use
from torchsummary import summary


class VAE_dense1(nn.Module):
    def __init__(self):
        super(VAE_dense1, self).__init__()

        size_oi = 300 # input/output / wavelet size
        size_inter = 100 # intermediate size
        size_lat = 5 # latent variable space size

        self.fc1 = nn.Linear(size_oi, size_inter)
        self.fc21 = nn.Linear(size_inter, size_lat)
        self.fc22 = nn.Linear(size_inter, size_lat)
        self.fc3 = nn.Linear(size_lat, size_inter)
        self.fc4 = nn.Linear(size_inter, size_oi)
        self.name = 'VAE_dense1'
        self.input_size = size_oi

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 300))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

class VAE_dense2(nn.Module):
    def __init__(self):
        super(VAE_dense2, self).__init__()

        size_oi = 280  # input/output / wavelet size
        size_inter = 100  # intermediate size
        size_lat = 5  # latent variable space size

        self.fc1 = nn.Linear(size_oi, size_inter)
        self.fc2 = nn.Linear(size_inter, size_inter)
        self.fc31 = nn.Linear(size_inter, size_lat)
        self.fc32 = nn.Linear(size_inter, size_lat)
        self.fc4 = nn.Linear(size_lat, size_inter)
        self.fc5 = nn.Linear(size_inter, size_inter)
        self.fc6 = nn.Linear(size_inter, size_oi)
        self.name = 'VAE_dense2'
        self.input_size = size_oi

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 280))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

class VAE_dense3(nn.Module):
    def __init__(self):
        super(VAE_dense3, self).__init__()

        size_oi = 280  # input/output / wavelet size
        size_inter = 100  # intermediate size
        size_lat = 5  # latent variable space size

        self.fc1 = nn.Linear(size_oi, size_inter)
        self.fc2 = nn.Linear(size_inter, size_inter)
        self.fc3 = nn.Linear(size_inter, size_inter)
        self.fc4 = nn.Linear(size_inter, size_inter)
        self.fc51 = nn.Linear(size_inter, size_lat)
        self.fc52 = nn.Linear(size_inter, size_lat)
        self.fc6 = nn.Linear(size_lat, size_inter)
        self.fc7 = nn.Linear(size_inter, size_inter)
        self.fc8 = nn.Linear(size_inter, size_inter)
        self.fc9 = nn.Linear(size_inter, size_inter)
        self.fc10 = nn.Linear(size_inter, size_oi)
        self.name = 'VAE_dense3'
        self.base_architecture = 'VAE_dense3'
        self.input_size = size_oi

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        return self.fc51(h4), self.fc52(h4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h6 = F.relu(self.fc6(z))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        h9 = F.relu(self.fc9(h8))
        return torch.sigmoid(self.fc10(h9))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 280))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    ################################




class VAE_conv1(nn.Module):
    def __init__(self):
        super(VAE_conv1, self).__init__()

        size_oi = 300 # input/output / wavelet size
        size_lat = 5 # latent variable space size

        self.cl1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=5,stride=2)
        self.cl2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.cl3 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=5,stride=2)

        self.ctl1 = nn.ConvTranspose1d(in_channels=32,out_channels=32,kernel_size=6,stride=2)
        self.ctl2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=6, stride=2)
        self.ctl3 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=6, stride=2)

        self.dese11 = nn.Linear(32*34, size_lat)
        self.dese12 = nn.Linear(32*34, size_lat)
        self.dense2 = nn.Linear(size_lat,32*34)
        self.name = 'VAE_conv1'
        self.input_size = size_oi

    def encode(self, x):
        x = x.reshape(256,1,self.input_size)
        h1 = F.relu(self.cl1(x))
        h2 = F.relu(self.cl2(h1))
        h3 = F.relu(self.cl3(h2)).reshape(256,-1)
        return self.dese11(h3), self.dese12(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.dense2(z)).reshape(256,32,-1)
        h5 = F.relu(self.ctl1(h4))
        h6 = F.relu(self.ctl2(h5))
        h7 = self.ctl3(h6).reshape(256,1,-1)
        return torch.sigmoid(h7)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 300))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class VAE_conv2(nn.Module):
    def __init__(self):
        super(VAE_conv2, self).__init__()

        size_oi = 280 # input/output / wavelet size
        size_lat = 5 # latent variable space size

        ks1 = 10
        ks2 = 10
        strides = 2

        n_filters = 128
        n_filters_small = 64
        self.n_filters = n_filters
        self.batch_size = 256

        self.cl1 = nn.Conv1d(in_channels=1,out_channels=n_filters_small,kernel_size=ks1,stride=strides)
        self.cl2 = nn.Conv1d(in_channels=n_filters_small, out_channels=n_filters_small, kernel_size=ks1, stride=strides)
        self.cl3 = nn.Conv1d(in_channels=n_filters_small,out_channels=n_filters,kernel_size=ks1,stride=strides)
        self.cl4 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=ks1, stride=strides)
        self.cl5 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=ks1, stride=strides)

        self.ctl1 = nn.ConvTranspose1d(in_channels=n_filters,out_channels=n_filters,kernel_size=ks2,stride=strides)
        self.ctl2 = nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=ks2, stride=strides)
        self.ctl3 = nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters_small, kernel_size=ks2, stride=strides)
        self.ctl4 = nn.ConvTranspose1d(in_channels=n_filters_small, out_channels=n_filters_small, kernel_size=ks2, stride=strides)
        self.ctl5 = nn.ConvTranspose1d(in_channels=n_filters_small, out_channels=1, kernel_size=ks2, stride=strides)

        self.dese11 = nn.Linear(n_filters, size_lat)
        self.dese12 = nn.Linear(n_filters, size_lat)
        self.dense2 = nn.Linear(size_lat,n_filters)
        self.name = 'VAE_conv2'
        self.input_size = size_oi

    def encode(self, x):
        x = x.reshape(self.batch_size,1,self.input_size)
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.cl3(x))
        x = F.relu(self.cl4(x))
        x = F.relu(self.cl5(x)).reshape(self.batch_size,-1)
        return self.dese11(x), self.dese12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.relu(self.dense2(z)).reshape(self.batch_size,self.n_filters,-1) # batch size, n_channels
        x = F.relu(self.ctl1(x))
        x = F.relu(self.ctl2(x))
        x = F.relu(self.ctl3(x))
        x = F.relu(self.ctl4(x))
        x = self.ctl5(x).reshape(self.batch_size,1,-1)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


# Mixed VAE Mixed conv input, dense output
class VAE_frank1(nn.Module):
    def __init__(self):
        super(VAE_frank1, self).__init__()

        size_oi = 280 # input/output / wavelet size
        size_lat = 5 # latent variable space size

        # conv parameters
        ks1 = 10
        strides = 2
        n_filters = 64
        n_filters_small = 32
        self.n_filters = n_filters
        self.batch_size = 256

        # dense parameters
        size_inter = 100

        self.cl1 = nn.Conv1d(in_channels=1,out_channels=n_filters_small,kernel_size=ks1,stride=strides)
        self.cl2 = nn.Conv1d(in_channels=n_filters_small, out_channels=n_filters_small, kernel_size=ks1, stride=strides)
        self.cl3 = nn.Conv1d(in_channels=n_filters_small,out_channels=n_filters,kernel_size=ks1,stride=strides)
        self.cl4 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=ks1, stride=strides)
        self.cl5 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=ks1, stride=strides)

        self.dese11 = nn.Linear(n_filters, size_lat)
        self.dese12 = nn.Linear(n_filters, size_lat)

        self.fc6 = nn.Linear(size_lat, size_inter)
        self.fc7 = nn.Linear(size_inter, size_inter)
        self.fc8 = nn.Linear(size_inter, size_inter)
        self.fc9 = nn.Linear(size_inter, size_inter)
        self.fc10 = nn.Linear(size_inter, size_oi)
        self.name = 'VAE_frank1'
        self.input_size = size_oi

        self.fc_clasi1 = nn.Linear(size_inter,size_inter)
        self.fc_clasi2 = nn.Linear(size_inter, 3)

    def encode(self, x):
        x = x.reshape(self.batch_size,1,self.input_size)
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.cl3(x))
        x = F.relu(self.cl4(x))
        x = F.relu(self.cl5(x)).reshape(self.batch_size,-1)
        return self.dese11(x), self.dese12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.relu(self.fc6(z))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        return torch.sigmoid(self.fc10(x))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def classify(self, x):
        x = x.reshape(self.batch_size, 1, self.input_size)
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.cl3(x))
        x = F.relu(self.cl4(x))
        x = F.relu(self.cl5(x)).reshape(self.batch_size, -1)
        x = F.relu(self.fc_clasi1(x))
        return F.softmax(self.fc_clasi2(x))

if __name__=='__main__':
    model = VAE_dense3()
    summary(model, input_size=(128,280))