import pandas as pd
from typing import NamedTuple
import json
import os

import sys
sys.path.append('..')
from model_and_training_configs_generator import get_model_and_training_configs

if sys.version_info[1]==6:
    from dataclasses import dataclass
else:
    print('you better be using python 3.7')


s3_files_path = 's3_local_pn.csv'


class FileMapper:

    def __init__(
            self,
            src_bucket = '',
            dst_bucket = '',
            profile_name ='',
            research_role =''
    ):

        self.src_bucket = src_bucket
        self.dst_bucket = dst_bucket
        self.profile_name = profile_name
        self.research_role = research_role


    def get_structured_files_df(self):
        df = get_model_and_training_configs()
        configs = df['config_name'].values.tolist()
        s3_config_paths = [f'path/{config}.h5' for config in configs]
        df['S3_CONFIG_FILE'] = s3_config_paths

        return df


    def create_s3_generator_function(self, n_cases=None):
        """
        Creates a generator function that iterates over the rows in the input df
        environment variable dicts. The dicts contain the following environment variables:
            S3_SRC:     This will contain the full S3 URL of each input h5.
            S3_REF:     This will contain the full S3 URL of each reference file (PSG / MySleep)
            S3_DEST:    This will be the output location passed in as a parameter.
        Args:
            none

        Retruns:
            A generator function that iterates over the S3 keys in the locations specified and returns dicts.
        """

        s3_file_df = self.get_structured_files_df()

        if n_cases is not None:
            s3_file_df = s3_file_df.iloc[0:n_cases,:]

        def _iter_func():
            for index, row in s3_file_df.iterrows():

                yield {
                    'environment': [
                        {
                            'name': 'S3_DATA_DIR',
                            'value': f"{row['s3_data_dir']}"
                        },
                        {
                            'name': 'S3_CONFIG_FILE',
                            'value': f"{row['S3_CONFIG_FILE']}"
                        },
                        {
                            'name': 'S3_MODEL_CASH',
                            'value': f"{row['s3_cash_location']}"
                        },
                    ],
                }

        return _iter_func


    def create_input_json(self):
        generator = self.create_s3_generator_function()
        dict_list = []
        for item in generator():
            entry_dict = {}
            env_lst = item['environment']
            for dict in env_lst:
                entry_dict[dict['name']] = dict['value']
            dict_list.append(entry_dict)
        with open('inputs.txt','w') as json_file:
            json.dump(dict_list, json_file)


if __name__ == '__main__':
    file_mapper = FileMapper()
    file_mapper.get_structured_files_df()
