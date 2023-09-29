# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import os
from collections import namedtuple
from data_prepper_config import  \
    Alignment, VaeEcgDataPrepCommonConfig, MitBihDataConfig, PtbDataConfig, StpDataConfig
from logging import debug
# import scipy.interpolate as interp
# import matplotlib as mpl
# mpl.use('MacOSX')


RecordTables = namedtuple('RecordTables', 'ecg rr beat_annotation rhythm_annotation')

class DataOptimizer:
    def __init__(self,file,beat_or_rhythm):
        self.config = VaeEcgDataPrepCommonConfig(span=1.2,window_size=300)
        self.study_map = {
            'mit-bih-ar':0,
            'mit-bih-af':1,
            'mit-bih-lt':2,
            'mit-bih-sv':3,
            'mit-bih-mv':4,
            'mit-bih-ns':5,
            'esc_st':6,
            'ptb_data':7,
            'st-petersburg-incart':8
        }
        self.found_acceptable_labels = False
        self.file = file
        self.beat_or_rhythm = beat_or_rhythm
        self.max_normal_ratio = 1

    def load_data(self):
        self.y_df = pd.read_hdf(self.file, 'y_df').reset_index()
        self.x_df = pd.read_hdf(self.file, 'x_df').reset_index()
        self.il_df = pd.read_hdf(self.file, 'int_l_df').reset_index()
        self.il_df.beat = self.il_df.beat.apply(lambda x: 0 if np.isnan(x) else x)
        self.il_df.rhythm = self.il_df.rhythm.apply(lambda x: 0 if np.isnan(x) else x)

    def get_preferred_leads(self):
        available_leads = self.y_df.leads[0].tolist()
        available_leads = [lead.upper() for lead in available_leads]
        preferred1 = ['ECG1', 'V5', 'MLII', 'V4', 'V6']
        preferred2 = ['ECG2', 'V2', 'V3']
        pref1_ix, pref2_ix = None, None
        for pref1 in preferred1:
            if pref1 in available_leads:
                pref1_ix = available_leads.index(pref1)
                break
        for pref2 in preferred2:
            if pref2 in available_leads:
                pref2_ix = available_leads.index(pref2)
                break
        if None in [pref1_ix, pref2_ix]:
            print('could not find good leads for current file}')
        else:
            self.pref1_ix = pref1_ix
            self.pref2_ix = pref2_ix
            self.found_acceptable_labels = True


    def balance_for_pathology(self):
        is_abnormal = self.il_df[self.beat_or_rhythm]>1
        is_normal = self.il_df[self.beat_or_rhythm]==1
        abnormal_ixs = self.il_df[is_abnormal].index
        normal_ixs = self.il_df[is_normal].index

        if normal_ixs.shape[0]>2*abnormal_ixs.shape[0]:
            normal_ixs = normal_ixs[:self.max_normal_ratio*abnormal_ixs.shape[0]]

        combined_ixs = normal_ixs.union(abnormal_ixs)
        self.il_df = self.il_df.loc[combined_ixs,:]
        self.y_df = self.y_df.loc[combined_ixs,:]
        self.x_df = self.x_df.loc[combined_ixs,:]



    def get_x_df_for_preferred_leads(self):
        x_df = self.x_df
        e1,e2 = self.pref1_ix, self.pref2_ix
        ws = self.config.window_size
        pref_cols = [f'ecg{e1:02}_{wi:03}' for wi in range(ws)] + [f'ecg{e2:02}_{wi:03}' for wi in range(ws)]
        renamed_cols = [f'ecg00_{wi:03}' for wi in range(ws)] + [f'ecg01_{wi:03}' for wi in range(ws)]
        rename_map = { old:new for old,new in zip(pref_cols,renamed_cols) }
        self.x_df = x_df[pref_cols].rename(columns=rename_map)


    def add_encoded_study_to_int_l_df(self):
        study_code = self.study_map[self.y_df.data_set[0]]
        self.il_df['study']=study_code

    def get_lx_df(self):
        return pd.concat([self.il_df,self.x_df])


    def optomize(self):
        self.load_data()
        self.get_preferred_leads()
        if self.found_acceptable_labels:
            self.get_x_df_for_preferred_leads()
            self.balance_for_pathology()
            self.add_encoded_study_to_int_l_df()
            self.get_lx_df()
        else:
            return None


class DataPrepper2:
    def __init__(self,common_conf,data_configs):
        self.common_conf = common_conf
        self.data_config = None
        self.data_configs =data_configs
        self.dfs_list = []

    def load_data_from_file(self, file_path):
        # Load file
        with DataStore(file_path, skip_validation=True) as store:
            rhythm_an = store.get_table_by_name('annotations/rhythm', display_warning=False)
            beat_an = store.get_table_by_name('annotations/beat', display_warning=False)
            ecg = store.get_table_by_name('ECG')
            rr = store.get_table_by_name('RR')
            if ecg is None:
                print(f'Could not load ecg from {file_path}')

        return RecordTables(ecg,rr,beat_an,rhythm_an)

    def remove_dc(self,record_data):
        if self.common_conf.remove_dc:
            rolling_window_size = self.common_conf.smoothing_window_in_seconds*self.data_config.fs
            ecg = record_data.ecg
            signal_median = ecg.rolling(window=rolling_window_size, center=True, min_periods=0, axis=0).median()
            ecg = ecg - signal_median
            return RecordTables(
                ecg=ecg,
                rr=record_data.rr,
                beat_annotation=record_data.beat_annotation,
                rhythm_annotation=record_data.rhythm_annotation
            )
        else:
            return record_data

    def scale_ecg(self, record_data):
        if self.common_conf.scale_ecg:
            rolling_window_size = self.common_conf.smoothing_window_in_seconds*self.data_config.fs
            ecg = record_data.ecg
            signal_std = ecg.rolling(window=rolling_window_size, center=True, min_periods=0, axis=0).std()
            ecg = ecg / signal_std
            return RecordTables(
                ecg=ecg,
                rr=record_data.rr,
                beat_annotation=record_data.beat_annotation,
                rhythm_annotation=record_data.rhythm_annotation
            )
        else:
            return record_data

    def resample_wavelets_df(self,wavelets):
        df_dict = {}
        for column, series in wavelets.iteritems():
            xp = np.array(range(series.size))
            fp = series.values
            x = np.linspace(xp[0], xp[-1], self.common_conf.window_size)
            y = np.interp(x, xp, fp)
            df_dict[column] = y

        df = pd.DataFrame(df_dict)
        return df

    def transform_record_to_table(self,record_data):
        ecg = record_data.ecg
        rr = record_data.rr
        beat_an = record_data.beat_annotation
        rhyth_an = record_data.rhythm_annotation

        if (ecg is None) or (rr is None):
            return None
        rr = rr.sort_index()
        ecg = ecg.sort_index()

        y_df = rr

        if beat_an is not None:
            y_df = y_df.merge(beat_an.sort_index(), how='left',left_index=True, right_index=True)

        if rhyth_an is not None:
            rhyth_an = rhyth_an\
                .sort_index()\
                .reindex(y_df.index,method='ffill')\
                .rename(columns={'original':'rhythm'})
            rhyth_an['rhythm'] = rhyth_an['rhythm'].str.replace('_START','').replace('_STOP','')
            y_df = y_df.merge(rhyth_an, how='left',left_index=True, right_index=True)

        src_window_size = self.common_conf.span*self.data_config.fs
        src_window_size = int(src_window_size//2)*2
        shoulder_size = int(src_window_size // 2)
        TableRow = namedtuple('TableRow', ['data_set', 'record', 'beat_ix', 'lead_name', 'rr', 'beat', 'rhythm'])
        x_df_list = []
        X = np.empty((y_df.shape[0]*ecg.shape[1], self.common_conf.window_size))
        X[:] = np.nan
        y_df_list = []
        length = y_df.shape[0]
        for beat_ix, (ix, y_df_row) in enumerate(y_df.iterrows()):
            print(f'\r{(beat_ix/length)*100:0.2f}% done', end='')
            ix_ecg = ecg.index.get_loc(ix, method='nearest')
            start_ix = ix_ecg - shoulder_size
            stop_ix = ix_ecg + shoulder_size
            if (start_ix >= 0) & (stop_ix < ecg.shape[0]):
                wavelets = ecg.iloc[start_ix:stop_ix, :]
                wavelets = self.resample_wavelets_df(wavelets)
                beat = [y_df_row.beat if 'beat' in y_df_row.index else None][0]
                rhythm = [y_df_row.rhythm if 'rhythm' in y_df_row.index else None][0]

                for lead_ix, (lead_name, column) in enumerate(wavelets.iteritems()):
                    y_df_list.append(TableRow(
                        data_set=self.data_set,
                        record=self.record,
                        beat_ix=beat_ix,
                        lead_name=lead_name,
                        rr = y_df_row.rr,
                        beat= beat,
                        rhythm=rhythm
                    ))
                    X[beat_ix*ecg.shape[1]+lead_ix,:] = column.values
        new_y_df = pd.DataFrame(y_df_list)
        x_df = pd.DataFrame(X, columns = self.common_conf.ecg_cols)
        df = pd.concat([new_y_df,x_df],sort=False,axis=1)
        return df

    def run(self):

        for data_config in self.data_configs:
            self.data_config = data_config
            temp_dir = self.data_config.outputdir
            os.makedirs(temp_dir, exist_ok=True)
            for folder_path in data_config.folder_paths:
                files = glob(os.path.join(folder_path,'*.h5'))
                for ix, file in enumerate(files):

                    self.record = os.path.basename(file).replace('.h5','')
                    self.data_set = os.path.basename(folder_path)

                    print(f'Processing {file} {100 * (ix / len(files)):.2f}%. doing {self.record}')
                    record_data = self.load_data_from_file(file)
                    record_data = self.remove_dc(record_data)
                    record_data = self.scale_ecg(record_data)
                    data_table = self.transform_record_to_table(record_data)

                    if data_table is not None:
                        table_path = os.path.join(temp_dir,f'{self.data_set}_{self.record}.h5')
                        data_table.to_hdf(table_path, 'df')

    def run_on_single_file(self,file_in, file_out):

        self.data_config = self.data_configs[0]
        record_data = self.load_data_from_file(file_in)
        record_data = self.remove_dc(record_data)
        record_data = self.scale_ecg(record_data)
        data_table = self.transform_record_to_table(record_data)

        if data_table is not None:
            os.makedirs(os.path.dirname(file_out), exist_ok=True)
            data_table.to_hdf(file_out, 'df')

    def leads_are_valid(self,df):
        lead_names = self.data_config.lead_names
        n_leads = self.data_config.number_of_leads
        # first check that the right number of leads are available, this is pretty critical
        if list(df['beat_ix'].value_counts().unique()) != [n_leads]:
            print(f'\n{df.data_set[0]} {df.record[0]} did not contain the correct number of leads!!\n')
            return False
        if lead_names is None:
            # use whatever columns were provided.
            return True
        if set(df['lead_name']) != set(self.data_config.lead_names):
            print(f'\n{df.data_set[0]} {df.record[0]} did not contain all the required lead names\n')
            return False
        return True

    #

    def concat_files_from_dir(self):

        x_columns = [f'ecg_{i:03}' for i in range(300)]

        for data_config in self.data_configs:
            dfs_list = []
            self.data_config = data_config
            temp_dir = self.data_config.outputdir
            files = glob(os.path.join(temp_dir,'*.h5'))

            for i, file in enumerate(files):
                df = pd.read_hdf(file,'df').dropna(axis = 0, subset=x_columns)

                if self.leads_are_valid(df):
                    print(f'\r{i/len(files):0.3} ...', end = '')
                    dfs_list.append(df)

            print('about to start saving')
            conc_df = pd.concat(dfs_list,sort=False,axis=0,ignore_index=True)
            conc_df = conc_df.dropna(axis = 0, subset=x_columns)
            # check that the n_rows is a multiple if n_leads
            assert (conc_df.shape[0] % self.data_config.number_of_leads == 0)
            name_out = f'{data_config.number_of_leads}lead__beats_0p6s_span.h5'
            conc_df.to_hdf(os.path.join(temp_dir,'concat',name_out),'df')

# hack to make ecg fs work without having it in the h5 file.
def get_config_from_s3_path(s3_path):
    from data_prepper_config import \
        MitBihMvDataConfig, MitBihSvDataConfig, MitBihArDataConfig, MitBihAfDataConfig, MitBihNsDataConfig, \
        PtbDataConfig, StpDataConfig, MitBihLtDataConfig
    configs = \
        [MitBihMvDataConfig(), MitBihSvDataConfig(), MitBihArDataConfig(), MitBihAfDataConfig(), MitBihNsDataConfig(),
        MitBihLtDataConfig(), PtbDataConfig(), StpDataConfig()]
    for config in configs:
        if config.id in s3_path:
            print(f'selected {config.id} with fs={config.fs} for {s3_path}')
            return config


def get_record_and_dataset_from_s3_path(s3_path):
    folders = s3_path.split('/')
    ds = None
    for folder in folders:
        if 'mit-bih-' in folder:
            ds = folder
        if 'ptb_data' in folder:
            ds = folder
        if 'st-petersburg-incart' in folder:
            ds = folder
    if ds is None:
        raise ValueError(f'could not identify the eataset name from s3 path for {s3_path}')
    rec = os.path.basename(s3_path).split('.')[0]
    return rec, ds


def run_formatter_on_file(file_in, file_out, s3_path,span):
    config = get_config_from_s3_path(s3_path)
    record, data_set = get_record_and_dataset_from_s3_path(s3_path)
    data_prepper = DataPrepper2(
        common_conf=VaeEcgDataPrepCommonConfig(span=float(span)),
        data_configs=[config]
    )
    data_prepper.record = record
    data_prepper.data_set = data_set
    data_prepper.run_on_single_file(file_in, file_out)


if __name__=='__main__':

    # data_prepper = DataPrepper2(
    #     common_conf = VaeEcgDataPrepCommonConfig(span = 0.6),
    #     data_configs = [
    #         # StpDataConfig(),
    #         # PtbDataConfig(),
    #         MitBihDataConfig(),
    #     ]
    # )
    # data_prepper.run()
    # data_prepper.concat_files_from_dir()

    # huge_hack...
    # parameters:
    # /Users/eduard/data/ecg/mit-bih-ar/100.h5
    # /Users/eduard/data/ecg_tables/ecg_2_lead/mit-bih-ar/derp_06.h5
    # 0.6
    # derp


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_in", help="path to the local copy of the h5 file")
    parser.add_argument("file_out", help="path to the output h5 file")
    parser.add_argument("s3_path", help='Path to either the input or output s3 location')
    parser.add_argument("span",help='The time span (window size), surrounding each beat.')
    args = parser.parse_args()

    run_formatter_on_file(args.file_in, args.file_out, args.s3_path, args.span)
