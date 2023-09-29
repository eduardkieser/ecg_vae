from __future__ import print_function
import argparse
import os
from glob import glob
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchsummary import summary
from ecg_loader_config import mixed_data_config,\
    is_afib_data_config,is_normal_data_config,is_other_data_config, ECGDataConfig
# use('MacOSX')
# from plotter import plot_training_graph, plot_sample_reconstructions
from plotter import Plotter
from autoencoders import  VAE_conv2, VAE_dense2, VAE_dense3, VAE_frank1
import gc
import numpy as np
from ecg_loader import AnnotatedECGDataset
from model_and_training_configs_generator import ModelAndTrainingConfig

# NB remove dc from ecg signal
# Create normalised, V1_centered representation
# experiment with removing KLD loss
# plot various cased to see it there is any clustering going on.

log_interval = 10
use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()

torch.manual_seed(1)

device = torch.device("cuda" if use_gpu else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

class SisoAutoEnTrainer:
    def __init__(self, model, config):

        self.model = model
        self.config = config
        self.reload_data()
        self.load_class_data()
        self.initialize_model_and_optimizer()

    def setup_model_dir(self):
        self.model_folder_path = f'models/{self.model.name}_' \
            f'{mixed_data_config.lead_name.value}_' \
            f'{mixed_data_config.window_alignment.value}'

        # create forder if it doesn't exist.
        os.makedirs(f'{self.model_folder_path}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.model_folder_path}/figures', exist_ok=True)


    def initialize_model_and_optimizer(self):
        self.train_losses = []
        self.test_losses = []
        self.learning_rates = []
        if self.model is not None:
            self.model = self.model.to(device)

            self.lr = 1e-3
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.setup_model_dir()
        if self.config.use_cached:
            self.load_state_dict_if_available()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def reload_data(self):
        self.train_loader = torch.utils.data.DataLoader(
            AnnotatedECGDataset(mixed_data_config,'training'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(
            AnnotatedECGDataset(mixed_data_config,'testing'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)


    def load_class_data(self):
        self.is_afib_loader = torch.utils.data.DataLoader(
            AnnotatedECGDataset(is_afib_data_config, model_mode='testing'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.is_normal_loader = torch.utils.data.DataLoader(
            AnnotatedECGDataset(is_normal_data_config, model_mode='testing'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.is_other_loader = torch.utils.data.DataLoader(
            AnnotatedECGDataset(is_other_data_config, model_mode='testing'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        self.afib_set = next(iter(self.is_afib_loader))
        self.normal_set = next(iter(self.is_normal_loader))
        self.other_set = next(iter(self.is_other_loader))

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, beat_pos=None,z=None):
        # My own little experiment, will try to force one of the latent space variables to coencide with beat_position
        # lets call it BeaTError

        BTE = F.mse_loss(z[:,0],beat_pos,reduction='sum')
        # BTE = 0
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.model.input_size), reduction='sum')
        MSE = F.mse_loss(recon_x, x.view(-1, self.model.input_size), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return MSE, KLD/20, BTE/10


    def train(self, epoch):
        self.model = self.model.to(device)
        self.model.train()
        train_loss = 0
        for batch_idx, (data) in enumerate(self.train_loader):
            x_data = data['wavelet'].to(device)
            beat_pos = data['beat_pos'].to(device)
            # print(f'train: data.shape {data.shape}')
            self.optimizer.zero_grad()
            recon_batch, mu, logvar,z = self.model(x_data)
            loss_ind = self.loss_function(recon_batch, x_data, mu, logvar, beat_pos=beat_pos,z=z)
            loss_print = [int(loss_ind[i].detach().numpy()) for i in range(3)]
            loss = sum(loss_ind)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx*len(x_data)}/{len(self.train_loader.dataset)} ({int(100* batch_idx / len(self.train_loader))}%)]\t'
                    f'Loss: {loss.item() / len(x_data):.2f}\t(MSE,KDL,BTE)={loss_print}')
        train_loss /= len(self.train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {train_loss}')
        return train_loss


    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data) in enumerate(self.test_loader):
                x_data = data['wavelet'].to(device)
                beat_pos = data['beat_pos'].to(device)
                # print(f'test: data.shape {data.shape}')
                recon_batch, mu, logvar, z = self.model(x_data)
                test_loss += sum(self.loss_function(recon_batch, x_data, mu, logvar, beat_pos=beat_pos,z=z)).item()
                if i == 0:

                    # ########### Plot reconstructed wavelets  ###############################
                    n = min(x_data.size(0), 8)
                    original_wavelets = []
                    reconstructed_wavelets = []
                    for rix in range(n):
                        original_wavelets.append(x_data[rix].cpu())
                        reconstructed_wavelets.append(recon_batch.view(self.config.batch_size, self.model.input_size)[rix].cpu().T)
                    Plotter(self).plot_wavelet_reconstruction(original_wavelets,reconstructed_wavelets,epoch,n)

                    # ########### plot latent space variables scatter plots #####################
                    afib_zs = self.model.reparameterize(*self.model.encode(self.afib_set['wavelet'])).detach().numpy()
                    normal_zs = self.model.reparameterize(*self.model.encode(self.normal_set['wavelet'])).detach().numpy()
                    other_zs = self.model.reparameterize(*self.model.encode(self.other_set['wavelet'])).detach().numpy()
                    Plotter(self).plot_latent_space(afib_zs,normal_zs,other_zs,epoch)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        gc.collect()
        return test_loss

    def save_state_dict(self,epoch, test_loss):
        PATH_LATEST = f'{self.model_folder_path}/checkpoints/state_dict_latest.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': test_loss,
            'train_losses': self.train_losses,
            'test_losses':  self.test_losses,
            'learning_rates': self.learning_rates
        }, PATH_LATEST)

    def load_state_dict_if_available(self):
        state_dict_path = f'{self.model_folder_path}/checkpoints/state_dict_latest.tar'
        if os.path.isfile(state_dict_path):
            checkpoint = torch.load(state_dict_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.epoch = epoch

            try:
                self.train_losses = checkpoint['train_losses']
                self.test_losses = checkpoint['test_losses']
                self.learning_rates = checkpoint['learning_rates']
                self.lr = self.learning_rates[-1]['learning_rate']
                print('loaded historical data')
            except:
                print(f'Could not load historical training data, might me loading from a older format checkpoint')
                self.lr = 0.0005

            print(f'Picking up from epoch: {epoch}, with a loss of: {loss}')

        self.model.to(device)


    def consider_reducing_lr(self):
        if self.epoch>0 and self.epoch%10==0:
            try:
                n_earlier = np.array(self.train_losses[-40:-20])
                n_later = np.array(self.train_losses[-20:])
                # half the lr if (∆mean < -(std()))
                delta = n_later.mean()-n_earlier.mean()  # should be a large negative number
                is_learning_well = delta < -n_later.std()
                if not is_learning_well:
                    self.lr = self.lr/2
                    print(f'reduced learning rate to {self.lr}')
                else:
                    print(f'{self.lr} still seems appropriate delta: {device} std: {n_later.std()}')
            except Exception as e:
                print(f'{e}, bouncing...')
            pass

    def fit(self):
        print(f'Training with Learning Rate: {self.optimizer.defaults["lr"]}')
        print('----------------------------------------------------------------')
        # train_losses = self.train_losses
        # test_losses = self.test_losses
        for epoch in range(self.epoch, self.epoch+1000):
            # self.reload_data()
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
            self.train_losses.append({'epoch':epoch, 'train_loss':train_loss})
            self.test_losses.append({'epoch':epoch, 'test_loss':test_loss})
            self.learning_rates.append({'epoch':epoch, 'learning_rate':self.lr})
            self.consider_reducing_lr()

            with torch.no_grad():
                sample = torch.randn(self.config.batch_size, 5).to(device).float()
                sample = self.model.decode(sample).cpu()
                Plotter(self).plot_sample_reconstructions(sample, epoch)
                Plotter(self).plot_training_graph(self.train_losses, self.test_losses, epoch)
                self.save_state_dict(epoch, test_loss)

    def summary(self):
        summary(self.model.float(), input_size=(128, 280))

if __name__ == "__main__":
    model = VAE_dense3()#.float()
    config = ModelAndTrainingConfig(
        limit_to=None,
        h5_path='/Users/eduard/workspaces/ml_projects/ecg_vae/ecg_vae/data/training_data/V1_center.h5',
        data_selection_mode=None
    )
    trainer = SisoAutoEnTrainer(model,config=config)
    trainer.summary()
    trainer.fit()

