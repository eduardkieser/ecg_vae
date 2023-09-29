from enum import Enum
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_prep.data_prepper_config import Alignment, LeadName
from typing import Any
# from model_and_training_configs_generator import DataSelectionMode

class ModelOperationMode(Enum):
    training = 'training'
    testing = 'testing'

class ECGDataConfig: # this is old
    def __init__(
            self,
            data_selection_mode: Any,
            size_oi: int,
            balance_col_list: list,
            limit_to: int,
            test_ratio: float,
            window_alignment: Alignment,
            lead_name: LeadName
    ):
        self.data_selection_mode = data_selection_mode
        self.size_oi = size_oi
        self.balance_col_list = balance_col_list
        self.limit_to = limit_to
        self.test_ratio = test_ratio
        self.window_alignment = window_alignment
        self.lead_name = lead_name
        self.h5_path = f'data/training_data/{self.lead_name.value}_{self.window_alignment.value}.h5'

limit_to = None

plot_limit = 256

window_alignment = Alignment.three_beat
lead_name = LeadName.V1


mixed_data_config = ECGDataConfig(
    data_selection_mode=None,
    size_oi=280,
    balance_col_list=['is_afib', 'is_normal', 'is_other'],
    limit_to=limit_to,
    test_ratio = 0.1,
    window_alignment=window_alignment,
    lead_name=lead_name
)
is_afib_data_config = ECGDataConfig(
    data_selection_mode='is_afib',
    size_oi=280,
    balance_col_list=['is_afib'],
    limit_to=plot_limit,
    test_ratio = 0.1,
    window_alignment=window_alignment,
    lead_name=lead_name
)
is_normal_data_config = ECGDataConfig(
    data_selection_mode='is_normal',
    size_oi=280,
    balance_col_list=['is_normal'],
    limit_to=plot_limit,
    test_ratio = 0.1,
    window_alignment=window_alignment,
    lead_name=lead_name
)
is_other_data_config = ECGDataConfig(
    data_selection_mode='is_other',
    size_oi=280,
    balance_col_list=['is_other'],
    limit_to=plot_limit,
    test_ratio = 0.1,
    window_alignment=window_alignment,
    lead_name=lead_name
)


# class VaeEcgDataProviderConfig_0p6_15_lead:
#     def __init__(
#             self,
#             h5_path = '/Users/eduard/data/ecg_tables/ecg_15_lead/15_lead_beats_0p6s_span.h5',
#             model_mode = 'training',
#             output_lead_names = ('v2','ii'),
#             input_lead_names = ('v2','ii'),
#             limit_to = None,
#             number_of_leads = 15
#             ):
#         self.h5_path = h5_path
#         self.input_lead_names = input_lead_names
#         self.output_lead_names = output_lead_names
#         self.model_mode = model_mode
#         self.data_selection_mode = None
#         self.test_ratio = 0.2
#         self.number_of_leads = number_of_leads
#         self.x_columns = [f'ecg_{i:03}'for i in range(300)]
#         self.limit_to = limit_to
#         if limit_to:
#             self.limit_to = (limit_to//number_of_leads)*number_of_leads
#
# class VaeEcgDataProviderConfig_0p6_12_lead:
#     def __init__(
#             self,
#             h5_path = '/Users/eduard/data/ecg_tables/ecg_12_lead/12_lead_beats_0p6s_span.h5',
#             model_mode = 'training',
#             output_lead_names = ('V2','II'),
#             input_lead_names = ('V2','II'),
#             limit_to = None,
#             number_of_leads = 12
#             ):
#         self.h5_path = h5_path
#         self.input_lead_names = input_lead_names
#         self.output_lead_names = output_lead_names
#         self.model_mode = model_mode
#         self.data_selection_mode = None
#         self.test_ratio = 0.2
#         self.number_of_leads = number_of_leads
#         self.x_columns = [f'ecg_{i:03}'for i in range(300)]
#         self.limit_to = limit_to
#         if limit_to:
#             self.limit_to = (limit_to//number_of_leads)*number_of_leads
#
#
# class VaeEcgDataProviderConfig_0p6_2_lead:
#     def __init__(
#             self,
#             h5_path = '/Users/eduard/data/ecg_tables/ecg_2_lead/mit-bih-af/04043.h5',
#             input_data_folder = '/Users/eduard/data/ecg_tables/ecg_2_lead/',
#             model_mode = 'training',
#             output_lead_names = None,
#             input_lead_names = None,
#             limit_to = None,
#             number_of_leads = 2
#             ):
#         self.h5_path = h5_path
#         self.input_lead_names = input_lead_names
#         self.output_lead_names = output_lead_names
#         self.model_mode = model_mode
#         self.data_selection_mode = None
#         self.test_ratio = 0.2
#         self.number_of_leads = number_of_leads
#         self.x_columns = [f'ecg_{i:03}'for i in range(300)]
#         self.limit_to = limit_to
#         self.input_data_folder = input_data_folder
#         if limit_to:
#             self.limit_to = (limit_to//number_of_leads)*number_of_leads

# class VaeEcgDataProviderConfig:
#     def __init__(self, config, model_mode = 'training'):
#         self.h5_path = config.h5_path
#         self.input_lead_names = config.input_lead_names
#         self.output_lead_names = config.output_lead_names
#         self.model_mode = model_mode
#         self.data_selection_mode = None
#         self.test_ratio = 0.2
#         self.number_of_leads = config.number_of_leads
#         self.x_columns = [f'ecg_{i:03}'for i in range(300)]
#         self.limit_to = limit_to
#         self.input_data_folder = config.input_data_folder
#         if config.limit_to:
#             self.limit_to = (config.limit_to//config.number_of_leads)*config.number_of_leads