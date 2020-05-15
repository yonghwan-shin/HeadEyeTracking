import FileHandling
import Analysing_functions
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from scipy import interpolate
from scipy import stats
from scipy import signal
import numpy as np
import statistics
from pathlib import Path
import pandas as pd
import demjson
import statsmodels.api as sm

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
    intp_holoX = interpolate.interp1d(holo.Timestamp, holo.HeadRotationX)
    intp_holoY = interpolate.interp1d(holo.Timestamp, holo.HeadRotationY)
    intp_holoPhi = interpolate.interp1d(holo.Timestamp, holo.Phi)
    intp_holoThe = interpolate.interp1d(holo.Timestamp, holo.Theta)
    eye = eye[eye['timestamp'] > 1]
    eye = eye[eye['confidence'] > 0.8]
    # removed_outliers = eye['norm_y'].between(eye['norm_y'].quantile(0.1), eye['norm_y'].quantile(0.9))
    # eye = eye.loc[removed_outliers]
    # fig, axs = plt.subplots(2, 1, figsize=[8, 8], sharex=False)
    if eye.shape[0] < 100: print(current_info, 'too short data');return
    filtered_y = Analysing_functions.butterworth_filter(eye.norm_y, fc=5)

    target_vertical = intp_holoX(eye.timestamp) + intp_holoThe(eye.timestamp)
    filtered_target_vertical = Analysing_functions.butterworth_filter(target_vertical, fc=5)
    peaks, _ = signal.find_peaks(filtered_target_vertical, distance=30)

    # axs[0].scatter(eye.timestamp, filtered_target_vertical, marker='+')
    peaks = np.append([0], peaks)
    peaks = np.append(peaks, [len(eye) - 1])
    # for peak in peaks:
        # axs[0].axvline(eye.timestamp.iloc[peak])
    for i in range(len(peaks) - 1):
        slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
            filtered_y[peaks[i]:peaks[i + 1]], filtered_target_vertical[peaks[i]:peaks[i + 1]])
        if slope_vertical > 0: print(i, 'weired regression');continue

        # axs[0].scatter(eye.timestamp.iloc[peaks[i]:peaks[i + 1]],
        #                (filtered_y[peaks[i]:peaks[i + 1]] * slope_vertical + intercept_vertical), marker='+')
        # axs[1].plot(filtered_y[peaks[i]:peaks[i + 1]],
        #             filtered_y[peaks[i]:peaks[i + 1]] * slope_vertical + intercept_vertical)
        # axs[1].scatter(filtered_y[peaks[i]:peaks[i + 1]], filtered_target_vertical[peaks[i]:peaks[i + 1]], marker='+',
        #                alpha=0.3)

        # sns.regplot(filtered_y,filtered_target_vertical,ax=axs[1],marker='+',scatter_kws={'color':'k','alpha':0.3})
        linregress_results.append(slope_vertical)
        print(i, slope_vertical, ':', r_vertical)

    print(current_info, 'drawn')
    # plt.show()
    pass


if __name__ == '__main__':
    subjects = [201]
    poss = ['W']
    envs = ['W']
    linregress_results = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = FileHandling.get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye = FileHandling.file_as_pandas(FileHandling.get_file_by_info(eye_file_list, current_info),
                                                  refined=True)
                holo = FileHandling.file_as_pandas(FileHandling.get_file_by_info(hololens_file_list, current_info))
                imu = FileHandling.file_as_pandas(FileHandling.get_file_by_info(imu_file_list, current_info))
                if eye.shape[0]<100: print('empty eye data');continue
                data_analysis(eye, holo, imu)
            except ValueError as err:
                print(err, current_info)
    plt.show()
    slopes = []
    for i in linregress_results:
        slopes.append(i)
    # plt.hist(slopes);
    sns.distplot(slopes,fit=stats.norm,kde=True)
    plt.show()
    mean_slope = sum(slopes) / len(slopes)
    print(mean_slope)
