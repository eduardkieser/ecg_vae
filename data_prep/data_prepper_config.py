from enum import Enum

class LeadName(Enum):
    MLII='MLII'
    V1='V1'
class Alignment(Enum):
    center = 'center'
    left = 'left'
    right = 'right'
    three_beat = 'three_beat'
    none = 'none'

# class EcgDataConfig:
#     def __init__(self):
#         self.lead_config = LeadName.MLII
#         self.alignment = Alignment.center
#         self.window_size = 300
#         self.folder_loc = '/Users/eduard/data/afib/MIT-BIH/'
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         self.one_hot_list = ['(N', '(AFIB', '(OTHER']
#         self.clean_ecg = True


# this is the config used by the data_prepper_v2 that is independednt of data set.
class VaeEcgDataPrepCommonConfig:
    def __init__(self,span, window_size=300):
        self.window_size = window_size # n samples
        self.span = span # seconds
        self.alignment = Alignment.center
        # TODO implement 0 median scalar (custom if necessary)
        # self.scalar = MinMaxScaler(feature_range=(0, 1))
        self.remove_dc = True
        self.scale_ecg = True
        self.smoothing_window_in_seconds = 3
        self.ecg_cols = [f'ecg_{n:03}' for n in range(self.window_size)]

# these are the configs used by the data_prepper_v2 that are specific to the data sets.
class MitBihDataConfig:
    def __init__(self):
        self.fs = 360 # sample rate (Hz)
        self.folder_paths = [
            # '/Users/eduard/data/ecg/mit-bih-mv', # 250 Hz
            # '/Users/eduard/data/ecg/mit-bih-sv', # 128 Hz
            # '/Users/eduard/data/ecg/mit-bih-ar', # 360 Hz
            # '/Users/eduard/data/ecg/mit-bih-af', # 250 Hz
            '/Users/eduard/data/ecg/mit-bih-ns' , # 128 Hz
        ]
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None
        ############################################
class MitBihMvDataConfig:
    def __init__(self):
        self.id = 'mit-bih-mv'
        self.fs = 250
        self.folder_paths = ['/Users/eduard/data/ecg/mit-bih-mv']
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None

class MitBihSvDataConfig:
    def __init__(self):
        self.id = 'mit-bih-sv'
        self.fs = 128
        self.folder_paths = ['/Users/eduard/data/ecg/mit-bih-sv']
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None

class MitBihArDataConfig:
    def __init__(self):
        self.id = 'mit-bih-ar'
        self.fs = 360
        self.folder_paths = ['/Users/eduard/data/ecg/mit-bih-ar']
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None

class MitBihAfDataConfig:
    def __init__(self):
        self.id = 'mit-bih-af'
        self.fs = 250
        self.folder_paths = ['/Users/eduard/data/ecg/mit-bih-af']
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None

class MitBihNsDataConfig:
    def __init__(self):
        self.id = 'mit-bih-ns'
        self.fs = 128
        self.folder_paths = ['/Users/eduard/data/ecg/mit-bih-ns']
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None

class MitBihLtDataConfig:
    def __init__(self):
        self.id = 'mit-bih-lt'
        self.fs = 128
        self.folder_paths = ['/Users/eduard/data/ecg/mit-bih-lt']
        self.number_of_leads = 2
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_2_lead'
        self.lead_names = None
        ############################################

class PtbDataConfig:
    def __init__(self):
        self.id = 'ptb_data'
        self.fs = 1000 # sample rate (Hz)
        self.folder_paths = ['/Users/eduard/data/ecg/ptb']
        self.number_of_leads = 15
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_15_lead'
        self.lead_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']

class StpDataConfig:
    def __init__(self):
        self.id='st-petersburg-incart'
        self.fs = 257 # sample rate (Hz)
        self.folder_paths = ['/Users/eduard/data/ecg/stp-incart']
        self.number_of_leads = 12
        self.outputdir = '/Users/eduard/data/ecg_tables/ecg_12_lead'
        self.lead_names = None

# these are the configs used by the vae_trainer and include model setup configs like ;operation mode

class ModelOperationMode(Enum):
    training = 'training'
    testing = 'testing'

leads_15= ('i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz')
leads_12= ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

