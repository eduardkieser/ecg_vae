from dataStore import DataStore
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import os
from collections import namedtuple
from data_prepper_config import EcgDataConfig, Alignment
from logging import debug
import scipy.interpolate as interp
import matplotlib as mpl
mpl.use('MacOSX')

class DataPrepper:
    def __init__(self,conf:EcgDataConfig):
        self.conf = conf
        self.y = []
        self.X = []

    def one_hot(self,y):
        categories = self.conf.one_hot_list
        return (y == np.array(categories)).astype(int)


    def load_data_from_file(self, file_path):
        # Load file
        with DataStore(file_path, skip_validation=True) as store:
            ryth_an = store.get_table_by_name('annotations/rhythm').sort_index()
            beat_an = store.get_table_by_name('annotations/beat').sort_index()
            ecg = store.get_table_by_name('ECG')[[self.conf.lead_config.value]]
            if ecg is None:
                print(f'we have a problem loading {self.conf.lead_config.value} from {file_path}')
        if self.conf.clean_ecg:
            rolling_window_size = 1024
            signal_median = ecg.rolling(window=rolling_window_size, center=True, min_periods=0).median()
            ecg = ecg-signal_median
        ryth_an = ryth_an.reindex(beat_an.index, method='ffill')
        y_df = pd.concat([beat_an, ryth_an], axis=1, sort=False)
        return y_df, ecg

    def get_y_X_from_file_path(self, file_path):
        try:
            y_df, ecg = self.load_data_from_file(file_path)
        except:
            IOError(f'failed to load signals from {file_path}')
            return False
        window_size = (self.conf.window_size//2)*2
        shoulder_size = window_size//2

        X = np.empty((y_df.shape[0], self.conf.window_size))
        X[:] = np.nan
        y = np.empty((y_df.shape[0],4))
        y[:]=np.nan

        # window is centered around the beat with no scaling
        if self.conf.alignment == Alignment.center:
            for x_ix, (ix, row) in enumerate(y_df.iterrows()):
                # find nearest ix in ecg
                # check that start_ix & end_ix is in ecg_signal
                ix_ecg = ecg.index.get_loc(ix, method='nearest')  # nearest_center_ix_in_ecg
                start_ix = ix_ecg - shoulder_size
                stop_ix = ix_ecg + shoulder_size
                if (start_ix >= 0) & (stop_ix < ecg.shape[0]):
                    wavelet = ecg.iloc[start_ix:stop_ix, 0].values.reshape(-1, 1)
                    norm_wavelet = self.conf.scaler.fit_transform(wavelet)
                    X[x_ix, :] = norm_wavelet.reshape(window_size)
                    y[x_ix, :3] = self.one_hot(row['classified'])
                    y[x_ix, 3] = 0.5

        # window spans from r peak of the previous beat to r peak of the next beat
        if self.conf.alignment == Alignment.three_beat:
            for x_ix, (ix, row) in enumerate(y_df.iterrows()):
                if (ix == y_df.index[0]) | (ix==y_df.index[-1]):
                    debug('skipping because first or last beat')
                    continue
                start_ix = ecg.index.get_loc(y_df.index[x_ix-1], method='nearest')
                stop_ix = ecg.index.get_loc(y_df.index[x_ix+1], method='nearest')
                center_ix = ecg.index.get_loc(y_df.index[x_ix], method='nearest')
                beat_pos = (center_ix-start_ix)/(stop_ix-start_ix)
                if (start_ix >= 0) & (stop_ix < ecg.shape[0]):
                    wavelet = ecg.iloc[start_ix:stop_ix, 0].values.reshape(-1, 1)
                    norm_wavelet = self.conf.scaler.fit_transform(wavelet)
                    norm_wavelet = pd.Series(norm_wavelet.flatten())
                    norm_wavelet.index = norm_wavelet.index * 1000
                    new_index = np.linspace(0, norm_wavelet.index[-1], window_size).astype(int)
                    norm_wavelet = norm_wavelet.reindex(new_index, method='nearest').values
                    X[x_ix, :] = norm_wavelet
                    y[x_ix, :3] = self.one_hot(row['classified'])
                    y[x_ix, 3] = beat_pos

        return y, X

    def get_y_X_from_directory(self):
        files = glob(f'{self.conf.folder_loc}*.h5')
        y_lst = []
        X_lst = []
        for ix,file in enumerate(files):
            print(f'working {file}')
            yX = self.get_y_X_from_file_path(file)
            if yX:
                y,X = yX
            else:
                continue
            y_lst.extend(y)
            X_lst.extend(X)

        self.y = np.vstack(y_lst)
        self.X = np.vstack(X_lst)

        _yX = np.hstack([self.y,self.X])
        columns = ['is_normal', 'is_afib', 'is_other','beat_pos'] + [str(i) for i in range(self.X.shape[1])]
        df = pd.DataFrame(_yX, columns=columns).dropna().astype({'is_normal': bool, 'is_afib': bool, 'is_other': bool})
        self.yX_df = df


    def save_training_data(self, prefix=''):
        directory = '../data/training_data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if len(self.y)!=0:
            self.yX_df.to_hdf(f'{directory}/{prefix}.h5', key='df',model='w')

    def load_training_data(self,prefix='', directory = '../training_data'):
        return pd.read_hdf(f'{directory}/{prefix}.h5',key='df')



if __name__=='__main__':
    prepper = DataPrepper(conf=EcgDataConfig())
    prepper.get_y_X_from_directory()
    prepper.save_training_data(prepper.conf.lead_config.value+'_'+prepper.conf.alignment.value)
    #
    # training_data = prepper.load_training_data('take1')
    # j=2
