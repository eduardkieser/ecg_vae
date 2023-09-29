import sys

sys.path.append('../')
# from ecg_loader_config import ECGDataConfig, DataSelectionMode, ModelOperationMode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import torch
from autoencoders import VAE_dense3

from matplotlib import use

use('MacOSX')

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_gpu else "cpu")



from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

blue = (3/255, 80/255, 160/255, 1)
orange = (252 / 255, 152 / 255, 3 / 255, 1)
green = (21 / 255, 158 / 255, 14 / 255, 1)

class WaveletFigure:
    def __init__(self):

        self.load_model()
        self.load_zs_df()


    def load_zs_df(self):
        zs_df_path = 'data/training_data/V1_center_with_zs.h5'
        self.zs_df = pd.read_hdf(zs_df_path,key='df')
        zs_columns = [f'z_{num}' for num in range(5)]
        wl_columns = [f'wl_{num}' for num in range(300)]
        label_columns = ['is_afib','is_normal','is_other']
        self.real_zs = self.zs_df[zs_columns].values.copy(order='C')
        self.real_wl = self.zs_df[wl_columns].values.copy(order='C')
        self.real_label = self.zs_df[label_columns].values.copy(order='C')


    def reverse_one_hot(self,labels):
        return ['is_afib', 'is_normal', 'is_other'][np.argmax(labels)]


    def load_model(self):
        state_dict_path = 'models/VAE_dense3_V1_center/checkpoints/state_dict_latest.tar'
        checkpoint = torch.load(state_dict_path)
        model = VAE_dense3()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(device).double()
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']

    def generate_wavelet(self, zs):
        zs = torch.DoubleTensor(np.array(zs))
        wavelet = self.model.decode(zs)
        return wavelet.detach().numpy()

    def get_closest_real_wavelet(self, zs):
        norms = np.linalg.norm(self.real_zs - zs, axis=1)
        row_ix = np.argmin(norms)
        # row_ixs = np.argpartition(norms, 5)
        # crws = self.real_wl[row_ixs,10:290]
        crw = self.real_wl[row_ix,10:290]
        label = self.real_label[row_ix,:]
        return crw, label

    def plot_space(self, ax):
        color_map = {0: blue, 1: green, 2: orange}
        y = self.zs_df.loc[:,['is_afib','is_normal','is_other']]
        y = y.iloc[:,1]*1+y.iloc[:,2]*2
        y = y.rename_axis('y')

        X_cols = [f'z_{num}' for num in range(5)]
        X = self.zs_df.loc[:,X_cols]

        model = LDA(n_components=2)
        # self.manifold = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000).fit(X)
        # self.manifold = MDS(n_components=2, max_iter=100).fit(X)
        model.fit(X,y)
        X2 = pd.DataFrame(model.transform(X))

        self.low_dim_df = pd.concat([y,X2],axis=1,ignore_index=True)
        self.low_dim_df.columns = ['y', '1', '2']
        self.low_dim_df = self.low_dim_df.sample(2000, axis=0)
        colors = self.low_dim_df.y.map(color_map)
        self.low_dim_df.plot(kind='scatter', x='1', y='2', c=colors,ax=ax,alpha=0.1,marker ='.')




    def make_figure(self):
        fig, ax = plt.subplots(1,2)
        plt.subplots_adjust(bottom=0.38)
        t = np.linspace(0.0, 1, 280)

        s = np.ones(len(t))
        l, = ax[0].plot(t, s, lw=2)
        l2, = ax[1].plot(t, s, lw=2)
        ax[0].margins(x=0)
        ax[0].set_ylim([-.1,1.1])
        ax[0].set_title('Reconstructed Wavelet')
        ax[1].margins(x=0)
        ax[1].set_ylim([-.1, 1.1])
        ax[1].set_title('Closest Real Wavelet')

        axz4 = plt.axes([0.1, 0.1, 0.8, 0.03]) # left bottom width height
        axz3 = plt.axes([0.1, 0.15, 0.8, 0.03])
        axz2 = plt.axes([0.1, 0.2, 0.8, 0.03])
        axz1 = plt.axes([0.1, 0.25, 0.8, 0.03])
        axz0 = plt.axes([0.1, 0.3, 0.8, 0.03])

        z0s = Slider(axz0, 'z0', -3, 3, valinit=0)
        z1s = Slider(axz1, 'z1', -3, 3, valinit=0)
        z2s = Slider(axz2, 'z2', -3, 3, valinit=0)
        z3s = Slider(axz3, 'z3', -3, 3, valinit=0)
        z4s = Slider(axz4, 'z4', -3, 3, valinit=0)


        def update(val):
            zs = [z0s.val, z1s.val, z2s.val, z3s.val, z4s.val]
            wavelet = self.generate_wavelet(zs)
            l.set_ydata(wavelet)
            fig.canvas.draw_idle()

            closest_wavelet, closest_label = self.get_closest_real_wavelet(zs)
            l2.set_ydata(closest_wavelet)
            colors_map = {
                'is_afib': (3/255, 80/255, 160/255, 1),
                'is_other':(252/255, 152/255, 3/255,1),
                'is_normal':(21/255, 158/255, 14/255,1)
            }
            closest_label = self.reverse_one_hot(closest_label)
            l2.set_color(colors_map[closest_label])
            ax[1].set_title(f'Closest Real Wavelet {closest_label}')
            fig.canvas.draw_idle()

        z0s.on_changed(update)
        z1s.on_changed(update)
        z2s.on_changed(update)
        z3s.on_changed(update)
        z4s.on_changed(update)

        update(0)
        plt.show()

if __name__=='__main__':

    WaveletFigure().make_figure()
