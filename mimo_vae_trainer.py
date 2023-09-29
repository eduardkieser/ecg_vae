from __future__ import print_function
import argparse
import os
from glob import glob
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchsummary import summary
from utils import sync_cash_to_s3
# from torch.utils.tensorboard import SummaryWriter

# use('MacOSX')
# from plotter import plot_training_graph, plot_sample_reconstructions
from plotter import Plotter
from mimo_autoencoders import VAE_mimo_dense
from ecg_loader import ECGDataProviderV2
from model_and_training_configs_generator import ModelAndTrainingConfig
import gc
import numpy as np

# NB remove dc from ecg signal
# Create normalised, V1_centered representation
# experiment with removing KLD loss
# plot various cased to see it there is any clustering going on.

log_interval = 10
use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()

torch.manual_seed(1)

device = torch.device("cuda" if use_gpu else "cpu")

print(f'using: {device} hardware for computation')

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

class MimoAutoEnTrainer:
    def __init__(
            self,
            model,
            config: ModelAndTrainingConfig,
    ):
        self.epoch = 0
        self.model = model
        self.config = config

        self.reload_data()
        self.initialize_model_and_optimizer()
        self.write_graph_to_logs()


    def setup_model_dir(self):
        self.model_folder_path = f'{self.config.local_cash_location}/{self.config.config_name}'

        # create forder if it doesn't exist.
        os.makedirs(f'{self.model_folder_path}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.model_folder_path}/figures', exist_ok=True)
        os.makedirs(f'{self.model_folder_path}/logs', exist_ok=True)

        # self.writer = SummaryWriter(f'{self.model_folder_path}/logs')


    def initialize_model_and_optimizer(self):
        self.train_losses = []
        self.test_losses = []
        self.learning_rates = []
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.double()
            self.lr = 1e-3
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.setup_model_dir()
        if self.config.use_cached:
            self.load_state_dict_if_available()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def write_graph_to_logs(self):
        # self.writer.add_graph(self.model, (torch.zeros[280,1],torch.zeros[280,1]))
        pass

    def reload_data(self):
        self.train_loader = torch.utils.data.DataLoader(
            ECGDataProviderV2(self.config, 'training'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(
            ECGDataProviderV2(self.config, 'testing'),
            batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        print(f'{len(self.train_loader)*self.config.batch_size} samples in training data')
        print(f'{len(self.test_loader)*self.config.batch_size} samples in testing data')


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, y0, y1, mu, logvar, z=None):
        # My own little experiment, will try to force one of the latent space variables to coencide with beat_position
        # lets call it BeaTError

        recon_x = torch.cat(recon_x,dim=1)
        y = torch.cat((y0,y1),dim=1)

        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.model.input_size), reduction='sum')
        MSE = F.mse_loss(recon_x, y, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        dxdt = recon_x[1:]-recon_x[:-1]
        dydt = y[1:]-y[:-1]

        MSE_dydt = F.mse_loss(dxdt,dydt, reduction='sum')

        return MSE, KLD/20, MSE_dydt


    def train(self, epoch):
        self.model = self.model.to(device)
        self.model.train()
        batch_size = self.config.batch_size
        train_loss = 0
        for batch_idx, (data) in enumerate(self.train_loader):
            x_data_0 = data['wavelet_in_0'].to(device)
            x_data_1 = data['wavelet_in_1'].to(device)

            y_data_0 = data['wavelet_out_0'].to(device)
            y_data_1 = data['wavelet_out_1'].to(device)

            self.optimizer.zero_grad()
            recon_batch, mu, logvar,z = self.model(x_data_0,x_data_1)
            loss_ind = self.loss_function(recon_batch, y_data_0, y_data_1, mu, logvar, z=z)
            loss_print = [int(loss_ind[i].detach().numpy()) for i in range(len(loss_ind))]
            loss = sum(loss_ind)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx*batch_size}/{len(self.train_loader)*batch_size} ({int(100* batch_idx / len(self.train_loader))}%)]\t'
                    f'Loss: {loss.item() / len(x_data_0):.2f}\t(MSE,KDL,MSE_ddt)={loss_print}')
        train_loss /= len(self.train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {train_loss}')
        return train_loss


    def iterate_through_data(self):
        for i, (data) in enumerate(self.test_loader):
            x_data_0 = data['wavelet_in_0'].to(device)
            x_data_1 = data['wavelet_in_1'].to(device)

            y_data_0 = data['wavelet_out_0'].to(device)
            y_data_1 = data['wavelet_out_1'].to(device)
            print(i)


    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data) in enumerate(self.test_loader):

                # data = {key:item.to(device) for key, item in data}
                x_data_0 = data['wavelet_in_0'].to(device)
                x_data_1 = data['wavelet_in_1'].to(device)

                y_data_0 = data['wavelet_out_0'].to(device)
                y_data_1 = data['wavelet_out_1'].to(device)

                # print(f'test: data.shape {data.shape}')
                recon_batch, mu, logvar, z = self.model(x_data_0,x_data_1)
                test_loss += sum(self.loss_function(recon_batch, y_data_0, y_data_1, mu, logvar, z=z)).item()
                if i == 0:

                    # ########### Plot reconstructed wavelets  ###############################
                    n = min(x_data_0.size(0), 8)
                    input_wavelets_0 = []
                    input_wavelets_1 = []
                    reference_wavelets_0 = []
                    reference_wavelets_1 = []
                    reconstructed_wavelets_0 = []
                    reconstructed_wavelets_1 = []
                    for rix in range(n):
                        input_wavelets_0.append((x_data_0[rix].cpu()))
                        input_wavelets_1.append((x_data_1[rix].cpu()))

                        reference_wavelets_0.append((y_data_0[rix].cpu()))
                        reference_wavelets_1.append((y_data_1[rix].cpu()))

                        reconstructed_wavelets_0.append(
                            recon_batch[0].view(self.config.batch_size, self.model.size_oi)[rix].cpu().T)
                        reconstructed_wavelets_1.append(
                            recon_batch[1].view(self.config.batch_size, self.model.size_oi)[rix].cpu().T)
                    wavelets = {
                        'input_wavelets':(input_wavelets_0,input_wavelets_1),
                        'reference_wavelets': (reference_wavelets_0, reference_wavelets_1),
                        'reconstructed_wavelets':(reconstructed_wavelets_0,reconstructed_wavelets_1)
                    }

                    Plotter(self).plot_mimo_wavelet_reconstruction(wavelets,epoch,n)

                    # ########### plot latent space variables scatter plots #####################
                    # afib_zs = self.model.reparameterize(*self.model.encode(self.afib_set['wavelet'])).detach().numpy()
                    # normal_zs = self.model.reparameterize(*self.model.encode(self.normal_set['wavelet'])).detach().numpy()
                    # other_zs = self.model.reparameterize(*self.model.encode(self.other_set['wavelet'])).detach().numpy()
                    # Plotter(self).plot_latent_space(afib_zs,normal_zs,other_zs,epoch)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        gc.collect()
        return test_loss

    def save_state_dict(self,epoch, test_loss):
        PATH_LATEST = f'{self.model_folder_path}/checkpoints/state_dict_latest.tar'
        print(f'saving to {PATH_LATEST}')
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
        if self.epoch>0 and self.epoch%20==0:
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

        # sync_cash_to_s3(self.config)

        for epoch in range(self.epoch, self.epoch+1000):
            # self.reload_data(int(epoch%20>=10))
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
            self.train_losses.append({'epoch':epoch, 'train_loss':train_loss})
            self.test_losses.append({'epoch':epoch, 'test_loss':test_loss})
            self.learning_rates.append({'epoch':epoch, 'learning_rate':self.lr})
            self.consider_reducing_lr()

            with torch.no_grad():
                # sample = torch.randn(self.batch_size, 5).to(device).double()
                # sample = (self.model.decode(sample)[0].cpu(), self.model.decode(sample)[0].cpu())
                # Plotter(self).plot_sample_reconstructions(sample, epoch)
                Plotter(self).plot_training_graph(self.train_losses, self.test_losses, epoch)
                self.save_state_dict(epoch, test_loss)

            sync_cash_to_s3(self.config)



    def summary(self):
        summary(self.model.float(), input_size=[(128, 280), (128, 280)])

if __name__ == "__main__":


    from model_and_training_configs_generator import ModelAndTrainingConfig
    config = ModelAndTrainingConfig()

    model = VAE_mimo_dense(size_i=50, size_g=70)
    trainer = MimoAutoEnTrainer(
        model=model,
        config=config,
    )
    trainer.fit()



