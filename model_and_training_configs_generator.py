from collections import namedtuple
from typing import NamedTuple, Any
import pandas as pd
import os
import shutil
from enum import Enum


# class DataSelectionMode(Enum):
#     mixed = None
#     is_afib = 'is_afib'
#     is_other = 'is_other'
#     is_normal = 'is_normal'

# input_data_folder = ''
# input_data_folder = '/Users/eduard/data/ecg_tables/ecg_2_lead'


x_columns: list = [f'ecg_{i:03}' for i in range(300)]
class ModelAndTrainingConfig(NamedTuple):
    # vae model parameters
    config_name: str = None
    base_architecture: str = None
    size_oi: int = 280
    size_g: int = 70
    size_i: int = 70
    size_z: int = 5
    n_individual_input_layers: int = 1
    n_grouped_input_layers: int = 1
    n_individual_output_layers: int = 1
    n_grouped_output_layers: int = 1
    # ecg_loading_parameters
    h5_path:str = None
    input_data_folder: str = '/Users/eduard/data/ecg_tables/ecg_2_lead'
    output_lead_names: tuple = None
    input_lead_names: tuple = None
    number_of_leads: int = 2
    limit_to: int = -1
    data_selection_mode: Any = None
    # vae training parameters
    epochs: int = 2
    s3_cash_location: str = 's3://eduard/ecg_vae/training_cache'
    local_cash_location: str = 'models_s3'
    s3_data_dir:str = 's3://eduard/ecg_beats_mini/span_0_6/'
    use_cached: bool = True
    batch_size: int = 256
    dynamic_lr: bool = True
    test_ratio: float = 0.2
    x_columns: list = x_columns
    balance_col_list: Any = ['is_afib', 'is_other', 'is_normal']

def get_model_and_training_configs():
    configs_list = [
        ModelAndTrainingConfig(config_name='mimo_dense_take1_z5', base_architecture='MIMO_VAE_DENSE'),
        ModelAndTrainingConfig(config_name='mimo_dense_take2_z2', base_architecture='MIMO_VAE_DENSE', size_z=2),
        ModelAndTrainingConfig(config_name='siso_dense_take2_z2', base_architecture='SISO_VAE_DENSE', size_z=2),
        ModelAndTrainingConfig(
            config_name='VAE_dense3_original_z5',
            base_architecture = 'VAE_dense3',
            limit_to=None,
            h5_path='/Users/eduard/workspaces/ml_projects/ecg_vae/ecg_vae/data/training_data/V1_center.h5',
            data_selection_mode=None
        )
    ]

    return pd.DataFrame(configs_list)

def create_config_files_from_config_df(df,output_dir='/Users/eduard/workspaces/ml_projects/ecg_vae/ecg_vae/DockerVaeTrainingCudaBase/batch_training_configs'):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    for ix, row in df.iterrows():
        row.to_hdf(os.path.join(output_dir,row['config_name']+'.h5'), 'df')

def prep_local_config_folder():
    df = get_model_and_training_configs()
    create_config_files_from_config_df(df)


if __name__ == '__main__':
    prep_local_config_folder()

