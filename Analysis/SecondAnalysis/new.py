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
import plotly.express as px

# import statsmodels.api as sm

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


def data_prep(eye, holo, imu):
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
    if eye.shape[0] < 100: print(current_info, 'too short data');return None
    filtered_y = Analysing_functions.butterworth_filter(eye.norm_y, fc=5)
    filtered_x = Analysing_functions.butterworth_filter(eye.norm_x, fc=5)
    target_vertical = intp_holoX(eye.timestamp) + intp_holoThe(eye.timestamp)
    target_horizontal = intp_holoY(eye.timestamp) - intp_holoPhi(eye.timestamp)
    filtered_target_vertical = Analysing_functions.butterworth_filter(target_vertical, fc=5)
    peaks, _ = signal.find_peaks(filtered_target_vertical, distance=30)

    peaks = np.append([0], peaks)
    peaks = np.append(peaks, [len(eye) - 1])

    eye['target_vertical'] = target_vertical
    eye['target_horizontal'] = target_horizontal
    eye['filtered_y'] = filtered_y
    eye['filtered_x'] = filtered_x
    return eye, holo, imu, peaks, filtered_x, filtered_y, target_vertical, target_horizontal


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
    filtered_x = Analysing_functions.butterworth_filter(eye.norm_x, fc=5)
    target_vertical = intp_holoX(eye.timestamp) + intp_holoThe(eye.timestamp)

    target_horizontal = intp_holoY(eye.timestamp) - intp_holoPhi(eye.timestamp)
    filtered_target_vertical = Analysing_functions.butterworth_filter(target_vertical, fc=5)
    peaks, _ = signal.find_peaks(filtered_target_vertical, distance=30)

    peaks = np.append([0], peaks)
    peaks = np.append(peaks, [len(eye) - 1])
    eye['target_vertical'] = target_vertical
    eye['target_horizontal'] = target_horizontal
    eye['filtered_y'] = filtered_y
    eye['filtered_x'] = filtered_x
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    sns.distplot(eye.target_vertical,fit=stats.norm,ax=axs1[0])
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    axs[1].axhline(0)
    axs[0].axhline(0)
    ss = []
    est=[]
    window_size = 40
    estimation_size = 5
    axs[0].plot(eye.timestamp, eye.target_vertical)
    axs[0].axvline(eye.timestamp.iloc[0])
    axs[0].axvline(eye.timestamp.iloc[window_size])
    for i in range(eye.shape[0] - window_size - estimation_size):
        y = eye.filtered_y[i:i + window_size]
        t = eye.target_vertical[i:i + window_size]
        timeline = eye.timestamp[i:i + window_size]
        s, itc, r, p, std = stats.linregress(y, t)

        estimation_timeline = eye.timestamp[i + window_size: i + window_size + estimation_size]
        estimation_y = eye.filtered_y[i + window_size:i + window_size + estimation_size]
        real_target = eye.target_vertical[i + window_size:i + window_size + estimation_size]
        estimation_target = s * estimation_y + itc
        # axs[0].scatter(timeline.iloc[-1], t.iloc[-1], marker='+', color='red')
        # axs[0].scatter(timeline.iloc[-1], y.iloc[-1] * s + i, marker='+', color='blue', alpha=0.3)  # -t.iloc[-1]
        axs[1].scatter(timeline.iloc[-1], s / 200, marker='+', color='black')
        axs[0].scatter(estimation_timeline, estimation_target, marker='+', color='red', alpha=0.3)
        axs[0].scatter(estimation_timeline, estimation_target - real_target, marker='x', color='black', alpha=0.3)
        est.append(estimation_target.iloc[0]-real_target.iloc[0])
        ss.append(s)
        # meansloe = sum(ss) / (len(ss))

    axs[1].axhline(sum(ss) / (len(ss) * 200))
    sns.distplot(est,fit=stats.norm,ax=axs1[1])
    plt.show()

    # for peak in peaks:
    # axs[0].axvline(eye.timestamp.iloc[peak])
    # for i in range(len(peaks) - 1):
    #     slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
    #         filtered_y[peaks[i]:peaks[i + 1]], filtered_target_vertical[peaks[i]:peaks[i + 1]])
    #     if slope_vertical > 0: print(i, 'weired regression');continue

    # axs[0].scatter(eye.timestamp.iloc[peaks[i]:peaks[i + 1]],
    #                (filtered_y[peaks[i]:peaks[i + 1]] * slope_vertical + intercept_vertical), marker='+')
    # axs[1].plot(filtered_y[peaks[i]:peaks[i + 1]],
    #             filtered_y[peaks[i]:peaks[i + 1]] * slope_vertical + intercept_vertical)
    # axs[1].scatter(filtered_y[peaks[i]:peaks[i + 1]], filtered_target_vertical[peaks[i]:peaks[i + 1]], marker='+',
    #                alpha=0.3)

    # sns.regplot(filtered_y,filtered_target_vertical,ax=axs[1],marker='+',scatter_kws={'color':'k','alpha':0.3})

    # linregress_results.append(slope_vertical)
    # print(i, slope_vertical, ':', r_vertical)
    # fig = px.scatter(data_frame=eye, x='target_horizontal', y='target_vertical', color='confidence',
    #                  animation_frame='timestamp')
    # fig.show()
    print(current_info, 'drawn')
    # plt.show()

    pass


def one_trial(subject, target, env, pos, block):
    [imu_file_list, eye_file_list, hololens_file_list] = FileHandling.get_one_subject_files(subject, refined=True)
    current_info = [target, env, pos, block]
    try:
        eye = FileHandling.file_as_pandas(FileHandling.get_file_by_info(eye_file_list, current_info),
                                          refined=True)
        holo = FileHandling.file_as_pandas(FileHandling.get_file_by_info(hololens_file_list, current_info))
        imu = FileHandling.file_as_pandas(FileHandling.get_file_by_info(imu_file_list, current_info))
        if eye.shape[0] < 100: print('empty eye data');return None
        return eye, holo, imu
        # data_analysis(eye, holo, imu)
    except ValueError as err:
        return None
        print(err, current_info)


if __name__ == '__main__':
    subjects = [201]
    poss = ['W']
    envs = ['U']
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
                if eye.shape[0] < 100: print('empty eye data');continue
                data_analysis(eye, holo, imu)
            except ValueError as err:
                print(err, current_info)
    plt.show()
    slopes = []
    for i in linregress_results:
        slopes.append(i)
    # plt.hist(slopes);
    sns.distplot(slopes, fit=stats.norm, kde=True)
    plt.show()
    mean_slope = sum(slopes) / len(slopes)
    print(mean_slope)
