import FileHandling
import Analysing_functions
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from scipy import interpolate
from scipy import stats
import numpy as np
import statistics
from pathlib import Path
import pandas as pd
import demjson

'''
options
'''
# subjects = range(201, 216)
subjects = range(201, 212)
# subjects=range(201,202)
targets = range(8)
envs = ['U', 'W']
# envs = ['U']
# poss = ['S', 'W']
poss = ['S']
# blocks=[0]
blocks = range(5)
pd.set_option('display.max_columns', 30)
ROOT = Path.cwd()
DATA_ROOT = ROOT.parent.parent / 'Datasets' / '2ndData'


# %% test
# file = DATA_ROOT /'201' /'EYE_T7_EW_PW_B3_C6_S201_0212172956.csv'
# eye = pd.read_csv(file,index_col=False,header=1)
def data_analysis(eye, holo, imu):
    eye.update({'timestamp': eye.timestamp - eye.timestamp.values[0] - 0.05})
    holo.update({'Timestamp': holo.Timestamp - holo.Timestamp.values[0]})
    rs = []
    thetas = []
    phis = []

    for index, row in holo.iterrows():
        x = row['TargetPositionX'] - row['HeadPositionX']
        y = row['TargetPositionY'] - row['HeadPositionY']
        z = row['TargetPositionZ'] - row['HeadPositionZ']
        [r, theta, phi] = Analysing_functions.asSpherical([x, z, y])
        rs.append(r)
        thetas.append(90 - theta)
        phis.append(90 - phi)

    holo['R'] = rs
    holo['Theta'] = thetas
    holo['Phi'] = phis

    eye = eye[(eye['timestamp'] > holo.Timestamp.iloc[0]) & (eye['timestamp'] < holo.Timestamp.iloc[-1])]
    # print('eye timestamp:', eye.timestamp.iloc[0], '->', eye.timestamp.iloc[-1])
    # print('holo timestamp:', holo.Timestamp.iloc[0], '->', holo.Timestamp.iloc[-1])
    intp_holoX = interpolate.interp1d(holo.Timestamp, holo.HeadRotationX)
    intp_holoY = interpolate.interp1d(holo.Timestamp, holo.HeadRotationY)
    intp_holoPhi = interpolate.interp1d(holo.Timestamp, holo.Phi)
    intp_holoThe = interpolate.interp1d(holo.Timestamp, holo.Theta)
    eye = eye[eye['timestamp']>1]
    eye = eye[eye['confidence']>0.8]
    removed_outliers = eye['norm_y'].between(eye['norm_y'].quantile(0.1),eye['norm_y'].quantile(0.9))
    eye = eye.loc[removed_outliers]
    # plt.scatter(eye.timestamp,eye.norm_y)
    # plt.show()
    # plt.scatter(eye.timestamp,intp_holoX(eye.timestamp)+intp_holoThe(eye.timestamp))
    # plt.show()
    sns.regplot(eye.norm_y,intp_holoX(eye.timestamp)+intp_holoThe(eye.timestamp),marker='+',scatter_kws={'color':'k','alpha':0.3})
    # plt.show()
    if eye['norm_y'].max()>1: print(current_info,'wrong value');return
    print(current_info,'drawn')
    pass


if __name__ == '__main__':
    # [imu_file_list, eye_file_list, hololens_file_list] = FileHandling.get_one_subject_files(203, refined=True)
    # current_info = [0, 'W', 'W', 3]
    # eye = FileHandling.file_as_pandas(FileHandling.get_file_by_info(eye_file_list, current_info), refined=True)
    # holo = FileHandling.file_as_pandas(FileHandling.get_file_by_info(hololens_file_list, current_info))
    # imu = FileHandling.file_as_pandas(FileHandling.get_file_by_info(imu_file_list, current_info))
    # data_analysis(eye, holo, imu)
    subjects=[201]
    poss = ['W']
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = FileHandling.get_one_subject_files(subject,refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye = FileHandling.file_as_pandas(FileHandling.get_file_by_info(eye_file_list, current_info),refined=True)
                holo = FileHandling.file_as_pandas(FileHandling.get_file_by_info(hololens_file_list,current_info))
                imu = FileHandling.file_as_pandas(FileHandling.get_file_by_info(imu_file_list,current_info))
                data_analysis(eye,holo,imu)
            except ValueError as err:
                print(err, current_info)
    plt.show()