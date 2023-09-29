from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_and_training_configs_generator import ModelAndTrainingConfig


from glob import glob
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AnnotatedECGDataset(Dataset):

    def __init__(self, config, model_mode):
        self.h5_path = config.h5_path
        self.data_raw = pd.read_hdf(config.h5_path, key='df').sample(frac=1,random_state=42)
        self.config = config
        self.model_mode = model_mode
        self.apply_selection()
        self.print_size()


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        label = torch.from_numpy(self.data.iloc[idx,:3].astype(np.int).values)
        beat_position = torch.from_numpy(self.data.iloc[idx,[3]].astype(np.float).values)
        wavelet = torch.from_numpy(self.data.iloc[idx,3:].astype(np.float).values)

        if self.config.size_oi:
            throw_away = wavelet.shape[0]-self.config.size_oi
            x0 = np.random.randint(throw_away)
            x1 = x0+self.config.size_oi
            wavelet = wavelet[x0:x1].float()
            beat_position = (beat_position*wavelet.shape[0]-x0)/(self.config.size_oi)
            beat_position = torch.tensor([beat_position]).float()
        sample = {'label':label, 'wavelet':wavelet,'beat_pos':beat_position}

        return sample

    def apply_selection(self):
        data_selection = self.config.data_selection_mode

        if data_selection!='all':

            if data_selection: # why was this here? God damn implicit shit! for filtering on conditino
                self.data_raw = self.data_raw[self.data_raw[data_selection]]

            l = self.data_raw.shape[0]
            if self.model_mode=='training':
                self.data_raw = self.data_raw.iloc[int(self.config.test_ratio*l):] # from test ratio to end
            else:
                self.data_raw = self.data_raw.iloc[:int(self.config.test_ratio*l)] # from start to test ratio

            if self.config.data_selection_mode is None:
                self.get_balanced_set()
            else:
                self.data=self.data_raw
        else:
            self.data = self.data_raw

        if self.config.limit_to:
            if self.data.shape[0]>self.config.limit_to:
                self.data = self.data.sample(self.config.limit_to,random_state=42)

        self.data_raw = None # clear some memory?

    def get_balanced_set(self):
        df = self.data_raw
        df = df.reset_index(drop=True)
        n_smallest_class = min([sum(df.loc[:, col]) for col in self.config.balance_col_list])
        df['gb_col'] = df.iloc[:, 0] + df.iloc[:, 1] * 2 + df.iloc[:, 2] * 3
        classes = []
        for ix, item in df.groupby('gb_col'):
            classes.append(item.sample(n_smallest_class))
        self.data = pd.concat(classes).sample(frac=1).drop('gb_col', axis=1)


    def print_size(self):

        print(f'Total of {self.data.shape[0]} beats! for {self.model_mode}')
        ms = self.data.memory_usage()
        ms = ms.values.sum() / 1e9
        print(f'df memory usage = {ms:0.2f} GB')


class ECGDataProviderV2(Dataset):

    def __init__(self,config, model_mode='training'):
        self.config = config
        self.model_mode = model_mode
        self.load_h5s()
        self.apply_train_test_selection()
        self.print_size()

    def load_h5s(self):
        x_columns = [f'ecg_{i:03}' for i in range(300)]
        dfs_list = []
        files = glob(os.path.join(self.config.input_data_folder, '*/*.h5'))
        files = files+glob(os.path.join(self.config.input_data_folder, '*.h5'))
        running_rows_total = 0

        for i, file in enumerate(files):
            print(f'loading {file}')
            df = pd.read_hdf(file, 'df').dropna(axis=0, subset=x_columns)
            if self.leads_are_valid(df):
                print(f'\r{i / len(files):0.3} ...', end='')
                dfs_list.append(df)
                running_rows_total+=df.shape[0]
                if running_rows_total>=self.config.limit_to:
                    print(f'{running_rows_total} (>{self.config.limit_to}) beats should to it')
                    break
        print('about to start concat')
        conc_df = pd.concat(dfs_list, sort=False, axis=0, ignore_index=True)
        conc_df = conc_df.dropna(axis=0, subset=x_columns)
        # check that the n_rows is a multiple if n_leads
        assert (conc_df.shape[0] % self.config.number_of_leads == 0)
        self.data = conc_df

    def leads_are_valid(self,df):
        lead_names = self.config.input_lead_names
        n_leads = self.config.number_of_leads
        # first check that the right number of leads are available, this is pretty critical
        if list(df['beat_ix'].value_counts().unique()) != [n_leads]:
            print(f'\n{df.data_set[0]} {df.record[0]} did not contain the correct number of leads!!\n')
            return False
        if lead_names is None:
            # number of beats per lead is in order, use whatever columns were provided.
            return True
        if set(df['lead_name']) != set(self.config.lead_names):
            print(f'\n{df.data_set[0]} {df.record[0]} did not contain all the required lead names\n')
            return False
        return True


    def print_size(self):
        print('Data Sets')
        print(self.data.data_set.value_counts())
        print(f'Total of {self.data.shape[0]} beats! for {self.model_mode}')
        ms = self.data.memory_usage()
        ms = ms.values.sum() / 1e9
        print(f'df memory usage = {ms:0.2f} GB')

    def apply_train_test_selection(self):
        n_leads = self.config.number_of_leads
        l = self.data.shape[0]//n_leads
        if self.model_mode == 'training':
            self.data = self.data.iloc[ int(self.config.test_ratio*l)*n_leads:]  # from test ratio to end
        elif self.model_mode == 'testing':
            self.data = self.data.iloc[:int(self.config.test_ratio*l)*n_leads ]  # from start to test ratio
        else:
            raise ValueError(f'model_mode should be either training or testing, we were given {self.model_mode}')

        assert (self.data.shape[0]%n_leads ==0)


    def __len__(self):
        l = int(self.data.shape[0]//self.config.number_of_leads)
        return l

    def __getitem__(self, idx):
        n_leads = self.config.number_of_leads
        ix = idx*n_leads
        sub_df = self.data.iloc[ix:ix+n_leads]
        # check that all the entries belong to the same beat
        if not (sub_df['beat_ix'].unique().shape[0]==1):
            print('fuckety fuck...')

        data_dict = {}
        # for ix, row in sub_df.iterrows():
        #     data_dict[row['lead_name']] = row[self.config.x_columns].values

        i0 = np.random.randint(0,20)
        i1 = i0+280

        wavelets = sub_df.sample(2)[self.config.x_columns].values
        wavelet_0 = wavelets[0, i0:i1]
        wavelet_1 = wavelets[1, i0:i1]

        columns = self.config.x_columns

        data_dict['wavelet_in_0'] = wavelet_0
        data_dict['wavelet_in_1'] = wavelet_1

        data_dict['wavelet_out_0'] = wavelet_0
        data_dict['wavelet_out_1'] = wavelet_1
        # except:
        #     print('fuck')

        return data_dict


if __name__=='__main__':

    config = ModelAndTrainingConfig(
        limit_to =None,
        h5_path='/Users/eduard/workspaces/ml_projects/ecg_vae/ecg_vae/data/training_data/V1_center.h5',
        data_selection_mode=None
    )
    ecgLoader = AnnotatedECGDataset(config, model_mode='training')
    for item in ecgLoader:
        print(item)
        break
