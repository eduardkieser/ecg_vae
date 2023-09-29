# from model_and_training_configs_generator import ModelAndTrainingConfig
from mimo_autoencoders import *
from siso_autoencoders import *
from mimo_vae_trainer import MimoAutoEnTrainer
from siso_vae_trainer import SisoAutoEnTrainer
from autoencoders import VAE_dense3
import pandas as pd
from utils import pull_cash_from_s3, pull_data_from_s3


def build_trainer_from_config(config, model):

    if config.base_architecture=='MIMO_VAE_DENSE':
        return MimoAutoEnTrainer(
            model=model,
            config=config,
        )
    if config.base_architecture in ['SISO_VAE_DENSE','VAE_dense3']:
        return SisoAutoEnTrainer(
            model=model,
            config=config,
        )


def build_VAE_from_config(config):

    if config.base_architecture == 'MIMO_VAE_DENSE':
        return VAE_mimo_dense(
            size_oi=config.size_oi,
            size_g=config.size_g,
            size_i=config.size_i,
            size_z=config.size_z,
            n_individual_input_layers=config.n_individual_input_layers,
            n_grouped_input_layers=config.n_grouped_input_layers,
            n_individual_output_layers=config.n_individual_output_layers,
            n_grouped_output_layers=config.n_grouped_output_layers,
            name=config.name,
        )
    if config.base_architecture == 'SISO_VAE_DENSE':
        return VAE_mimo_dense(
            size_oi=config.size_oi,
            size_g=config.size_g,
            size_z=config.size_z,
            n_grouped_input_layers=config.n_grouped_input_layers,
            n_grouped_output_layers=config.n_grouped_output_layers,
            name=config.name,
        )
    if config.base_architecture=='VAE_dense3':
        return VAE_dense3()


def run(confi_file, data_dir, local_model_cash):
    print('Running with args:')
    print(f'-> confi_file: {confi_file}')
    print(f'-> data_dir: {data_dir}')
    print(f'-> model_cash: {local_model_cash}')

    config = pd.read_hdf(confi_file)
    config.input_data_folder = data_dir
    config.local_cash_location = local_model_cash

    pull_cash_from_s3(config)
    pull_data_from_s3(config)

    model = build_VAE_from_config(config).float()
    trainer = build_trainer_from_config(config, model)
    print('running fit')
    trainer.fit()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to .h5 config_file")
    parser.add_argument("data_dir", help="path to training data")
    parser.add_argument("local_model_cash", help="path to .h5 config_file")
    # parser.add_argument("s3_model_cash_loc", help="path to .h5 config_file")
    args = parser.parse_args()

    run(args.config_file, args.data_dir, args.local_model_cash)

# Local configs:
# /Users/eduard/workspaces/ml_projects/ecg_vae/ecg_vae/DockerVaeTrainingCudaBase/batch_training_configs/mimo_dense_take1_z5.h5
# /Users/eduard/data/ecg_tables/ecg_2_lead
# /Users/eduard/workspaces/ml_projects/ecg_vae/ecg_vae/models
