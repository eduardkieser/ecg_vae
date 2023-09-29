import subprocess
import os
import shutil

def sync_cash_to_s3(config):
    print("PUSHING CASH...")
    s3_cash_path = os.path.join(config.s3_cash_location, config.config_name)
    synch_string = f'aws s3 sync {config.local_cash_location} {s3_cash_path}'
    print(synch_string)
    subprocess.run(synch_string.split(' '))

def pull_cash_from_s3(config):
    print("PULLING CASH...")
    os.makedirs(config.local_cash_location, exist_ok=True)
    s3_cash_path = os.path.join(config.s3_cash_location,config.config_name)
    sync_string = f'aws s3 sync {s3_cash_path} {config.local_cash_location}'
    print(sync_string)
    subprocess.run(sync_string.split(' '))

def pull_data_from_s3(config):
    print("PULLING DATA...")
    exclude_string = '--exclude "*st-petersburg*" --exclude "*ptb_data*" --exclude "*mit-bih-af* --exclude "*mit-bih-lt*"'
    sync_string = f'aws s3 sync {config.s3_data_dir} {config.input_data_folder} {exclude_string}'
    print(sync_string)
    subprocess.run(sync_string.split(' '))

    total, used, free = shutil.disk_usage("/")

    print("Total: %d GiB" % (total // (2 ** 30)))
    print("Used: %d GiB" % (used // (2 ** 30)))
    print("Free: %d GiB" % (free // (2 ** 30)))

