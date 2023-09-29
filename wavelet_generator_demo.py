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


class WaveletFigure:
    def __init__(self):

        self.load_model()


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

    def make_figure(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.38)
        t = np.linspace(0.0, 1, 280)

        s = np.ones(len(t))
        l, = plt.plot(t, s, lw=2)
        ax.margins(x=0)
        ax.set_ylim([-.1,1.1])
        ax.set_title('Wavelet')

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

        z0s.on_changed(update)
        z1s.on_changed(update)
        z2s.on_changed(update)
        z3s.on_changed(update)
        z4s.on_changed(update)

        update(0)
        plt.show()


if __name__=='__main__':
    WaveletFigure().make_figure()