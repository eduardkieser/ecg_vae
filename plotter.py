from matplotlib import pyplot as plt
# from main_vae_ecg import AutoEnTrainer

class Plotter:

    def __init__(self, trainer):
        self.trainer =trainer


    def plot_training_graph(self, train_losses, test_losses, epoch):

        # make backwards compatable, if if list of float no, problem, if list of dict, do something
        if type(train_losses[0]) is dict: #i.e. new structure
            # print('converting from new structure')
            try:
                train_losses = [tli['train_loss'] for tli in train_losses]
                test_losses = [tli['test_loss'] for tli in test_losses]
            except:
                print('got weird problems failing, bouncing')
                return

        fig, ax = plt.subplots()
        ax.plot(train_losses, label='train losses')
        ax.plot(test_losses, label='test losses')
        ax.legend()
        fig.savefig(f'{self.trainer.model_folder_path}/figures/train_graph_all.png')
        ax.set_title(f'Training graph at epoch{epoch}')
        if len(train_losses) > 40:
            train_losses_sub = train_losses[-40:]
            test_losses_sub = test_losses[-40:]
        else:
            train_losses_sub = train_losses
            test_losses_sub = test_losses

        if len(train_losses) > 10:
            train_losses_10 = train_losses[-10:]
            test_losses_10 = test_losses[-10:]
        else:
            train_losses_10 = train_losses
            test_losses_10 = test_losses

        fig, ax = plt.subplots()
        ax.plot(train_losses_sub, label='train losses')
        ax.plot(test_losses_sub, label='test losses')
        ax.legend()
        fig.savefig(f'{self.trainer.model_folder_path}/figures/train_graph_latest.png')
        fig, ax = plt.subplots()
        ax.plot(train_losses_10, label='train losses')
        ax.plot(test_losses_10, label='test losses')
        ax.legend()
        fig.savefig(f'{self.trainer.model_folder_path}/figures/train_graph_latest_10.png')


    def plot_sample_reconstructions(self,sample,epoch):
        fig, ax = plt.subplots(8,2, sharey=True)
        ax[0,0].set_title('Random Sampled Wavelets')
        for rix in range(8):
            ax[rix, 0].plot(sample[rix])
            ax[rix, 1].plot(sample[rix+8])
        fig.savefig(f'{self.trainer.model_folder_path}/figures/sample_latest.png')
        if epoch%20==0:
            fig.savefig(f'{self.trainer.model_folder_path}/figures/sample_{epoch}.png')


    def plot_wavelet_reconstruction(self,original_wavelets: list, reconstructed_wavelets:list,epoch,n):
        folder_path = self.trainer.model_folder_path
        fig, ax = plt.subplots(n, 2, sharey=True)
        for rix in range(n):
            ax[rix, 0].plot(original_wavelets[rix])
            ax[rix, 1].plot(reconstructed_wavelets[rix])

        ax[0, 0].set_title('Original Wavelets')
        ax[0, 1].set_title('Reconstructed Wavelets')
        fig.savefig(f'{folder_path}/figures/reconstruction_latest.png')
        if epoch%20==0:
            fig.savefig(f'{folder_path}/figures/reconstruction_{epoch}.png')


    def plot_mimo_wavelet_reconstruction(self,wavelets:dict, epoch:int, n=8):
        folder_path = self.trainer.model_folder_path
        fig, ax = plt.subplots(n, 3, sharey=True)
        for rix in range(n):
            i0, i1 = wavelets['input_wavelets']
            ax[rix, 0].plot(i0[rix])
            ax[rix, 0].plot(i1[rix])

            r0, r1 = wavelets['reference_wavelets']
            ax[rix, 1].plot(r0[rix])
            ax[rix, 1].plot(r1[rix])

            o0, o1 = wavelets['reconstructed_wavelets']
            ax[rix, 2].plot(o0[rix])
            ax[rix, 2].plot(o1[rix])
        ax[0, 0].set_title('Input Wavelets')
        ax[0, 1].set_title('Reference Wavelets')
        ax[0, 2].set_title('Reconstructed Wavelets')
        fig.savefig(f'{folder_path}/figures/reconstruction_latest.png')
        if epoch%20==0:
            fig.savefig(f'{folder_path}/figures/reconstruction_{epoch}.png')




    def plot_latent_space(self,afib_zs,normal_zs,other_zs,epoch):

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        alpha = 0.3
        for ix, axix in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax[axix].scatter(afib_zs[:, ix], afib_zs[:, ix + 1], alpha=alpha, label='afib')
            ax[axix].scatter(other_zs[:, ix], other_zs[:, ix + 1], alpha=alpha, label='other')
            ax[axix].scatter(normal_zs[:, ix], normal_zs[:, ix + 1], alpha=alpha, label='normal')
            ax[axix].legend()
            ax[axix].set_title(f'latens space distrobutions z_{ix} vs z_{ix + 1}')
        fig.savefig(f'{self.trainer.model_folder_path}/figures/z_scatter_latest.png')
        if epoch%20==0:
            fig.savefig(f'{self.trainer.model_folder_path}/figures/z_scatter_{epoch}.png')