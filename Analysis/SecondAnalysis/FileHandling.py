import os.path
from pathlib import Path
import pprint
import pandas as pd

pd.set_option('display.max_columns', 30)

dataset_folder_name = "2ndData"

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT.parent.parent / 'Datasets' / dataset_folder_name
print('DATA ROOT:', DATA_ROOT)
pp = pprint.PrettyPrinter(indent=4)


# example: EYE_T0_EU_PW_B2_C7_S7_0206164759.csv
def get_file_by_info(_file_list, _info):
    for file in _file_list:
        if make_trial_info(_info) in file.name:
            # print('getting', make_trial_info(_info), file.name)
            return file
    return None


def make_trial_info(info):
    target = info[0]
    env = info[1]
    pos = info[2]
    block = info[3]
    # c = info[4]
    # sub_num = info[5]
    output = 'T' + str(target) + "_E" + str(env) + '_P' + str(pos) + '_B' + str(block)
    return output


def get_trial_info(_file_name):
    output = []
    try:
        file_data = _file_name.split('_')
        target = file_data[1][1:]
        env = file_data[2][1:]
        pos = file_data[3][1:]
        block = file_data[4][1:]
        c = file_data[5][1:]
        sub_num = file_data[6][1:]
        output = [target, env, pos, block]
    except ValueError as err:
        print(err.args)
    return output


def get_one_subject_files(_sub_num):
    subject_folder = DATA_ROOT / str(_sub_num)
    hololens_folder = DATA_ROOT / 'hololens_data' / ('compressed_sub' + str(_sub_num))
    eye_file_list = []
    imu_file_list = []
    hololens_file_list = []
    eye_files = subject_folder.rglob('EYE*.csv')
    imu_files = subject_folder.rglob('IMU*.csv')
    hololens_files = hololens_folder.rglob('*.csv')
    for file in eye_files:
        eye_file_list.append(file)
    for file in imu_files:
        imu_file_list.append(file)
    for file in hololens_files:
        hololens_file_list.append(file)
    return [imu_file_list, eye_file_list, hololens_file_list]


def file_as_pandas(_file_path):
    if _file_path.exists() and _file_path.is_file():
        dataframe = pd.read_csv(_file_path, index_col=False, header=1)
        return dataframe


def main():
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(6)
    # pprint.pprint(hololens_file_list)
    ss = get_file_by_info(hololens_file_list, [0, 'U', 'S', 1])


if __name__ == "__main__":
    main()
