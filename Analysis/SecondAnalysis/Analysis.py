from FileHandling import *
from DataManipulation import *
from Analysing_functions import *
import matplotlib.pyplot as plt
import itertools
from scipy import interpolate
from scipy.signal import find_peaks
import numpy as np
import torch
import statistics
import signal_detection

# subjects = range(201, 216)
subjects = range(201, 212)
# subjects=range(201,202)
targets = range(8)
envs = ['U', 'W']
poss = ['S', 'W']
# poss = ['W']
blocks = range(5)


def create_eye_csv():
    # To reduce parsing json time...
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye_data = get_file_by_info(eye_file_list, current_info)
                df_eye = manipulate_eye(file_as_pandas(eye_data))

                norm_x = []
                norm_y = []
                for row in df_eye.itertuples():
                    norm_x.append(float(row[15][0]))
                    norm_y.append(float(row[15][1]))
                df_eye['norm_x'] = norm_x
                df_eye['norm_y'] = norm_y

                print(df_eye.head(1)['norm_x'])
                df_eye.to_csv(path_or_buf=(DATA_ROOT / 'refined_eye_data' / ('refined_' + eye_data.name)), index=False)
                print('saved', eye_data.name, '  ', current_info, subject)
            except ValueError as err:
                print(err, current_info)


def summary_holo_data_proceed():
    df_holo = pd.read_csv(DATA_ROOT / 'hololens_analysis' / 'summary_1st_analysis.csv')
    output = []
    df_output = pd.DataFrame(columns=df_holo.columns)
    df_WW = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'W')]
    df_UW = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'W')]
    df_US = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'S')]
    df_WS = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'S')]
    WW = df_WW.mean()
    WW['env'] = 'W'
    WW['pos'] = 'W'
    UW = df_UW.mean()
    UW['env'] = 'U'
    UW['pos'] = 'W'
    US = df_US.mean()
    US['env'] = 'U'
    US['pos'] = 'S'
    WS = df_WS.mean()
    WS['env'] = 'W'
    WS['pos'] = 'S'

    std_WW = df_WW.std()

    std_UW = df_UW.std()
    std_US = df_US.std()
    std_WS = df_WS.std()

    std_WW['subject'] = ''
    std_WW['target'] = ''
    std_WW['env'] = ''
    std_WW['pos'] = 'SD:'
    std_UW['subject'] = ''
    std_UW['target'] = ''
    std_UW['env'] = ''
    std_UW['pos'] = 'SD:'

    std_US['subject'] = ''
    std_US['target'] = ''
    std_US['env'] = ''
    std_US['pos'] = 'SD:'
    std_WS['subject'] = ''
    std_WS['target'] = ''
    std_WS['env'] = ''
    std_WS['pos'] = 'SD:'

    df_output = df_output.append(WW, ignore_index=True)
    df_output = df_output.append(std_WW, ignore_index=True)
    df_output = df_output.append(UW, ignore_index=True)
    df_output = df_output.append(std_UW, ignore_index=True)
    df_output = df_output.append(US, ignore_index=True)
    df_output = df_output.append(std_US, ignore_index=True)
    df_output = df_output.append(WS, ignore_index=True)
    df_output = df_output.append(std_WS, ignore_index=True)
    df_output.to_csv(path_or_buf=(DATA_ROOT / 'hololens_analysis' / 'summary_1st_analysis_proceed.csv'), index=False)


def summary_holo_data():
    df_holo = pd.read_csv(DATA_ROOT / 'hololens_analysis' / '1st_analysis.csv')
    output = []
    df_output = pd.DataFrame(columns=df_holo.columns)
    for subject in subjects:
        df_subject = df_holo[df_holo['subject'] == subject]
        df_WW = df_subject[(df_subject['env'] == 'W') & (df_subject['pos'] == 'W')]
        df_UW = df_subject[(df_subject['env'] == 'U') & (df_subject['pos'] == 'W')]
        df_US = df_subject[(df_subject['env'] == 'U') & (df_subject['pos'] == 'S')]
        df_WS = df_subject[(df_subject['env'] == 'W') & (df_subject['pos'] == 'S')]
        # print(df_subject.shape[0], df_WW.shape[0], df_UW.shape[0], df_US.shape[0], df_WS.shape[0])

        WW = df_WW.mean()
        WW['env'] = 'W'
        WW['pos'] = 'W'
        UW = df_UW.mean()
        UW['env'] = 'U'
        UW['pos'] = 'W'
        US = df_US.mean()
        US['env'] = 'U'
        US['pos'] = 'S'
        WS = df_WS.mean()
        WS['env'] = 'W'
        WS['pos'] = 'S'
        df_output = df_output.append(WW,ignore_index=True)
        df_output = df_output.append(UW,ignore_index=True)
        df_output = df_output.append(US,ignore_index=True)
        df_output = df_output.append(WS,ignore_index=True)

        # df_output.append(df_WW.mean(),ignore_index=True)
        # df_WW = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'W')]
        # df_UW = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'W')]
        # df_US = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'S')]
        # df_WS = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'S')]
        # print(df_holo.shape[0], df_WW.shape[0], df_UW.shape[0], df_US.shape[0], df_WS.shape[0])

    df_output.to_csv(path_or_buf=(DATA_ROOT / 'hololens_analysis' / 'summary_1st_analysis.csv'), index=False)
    print('saved')


def analyse_holo_data():
    outputList = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                print(subject, current_info)
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_holo = file_as_pandas(hololens_data)

                check_holo_file(df_holo, current_info)

                target_name = 'target_' + str(target)
                _init_time = df_holo.head(1)['Timestamp'].values[0]
                _end_time = df_holo.tail(1)['Timestamp'].values[0]
                total_time = _end_time - _init_time
                frame_rate = df_holo.shape[0] / total_time

                list_target_in = []
                list_target_out = []
                list_duration_in = []
                list_duration_out = []
                previous_target = ''
                for row in df_holo.itertuples(index=False):
                    if row[13] == target_name and previous_target != target_name:
                        list_target_in.append(row[0])
                    if row[13] != target_name and previous_target == target_name:
                        list_target_out.append(row[0])
                    previous_target = row[13]

                num_over = len(list_target_in)
                num_off = len(list_target_out)
                if num_over < 1:
                    print('zero over count')
                    continue

                start_time = list_target_in[0] - _init_time
                if num_over == num_off:
                    for in_count in range(num_off):
                        list_duration_in.append(list_target_out[in_count] - list_target_in[in_count])
                elif num_over == num_off + 1:
                    for in_count in range(num_off):
                        list_duration_in.append(list_target_out[in_count] - list_target_in[in_count])
                    list_duration_in.append(_end_time - list_target_in[-1])
                else:
                    print('error in matching in n out')
                    continue
                df_trial = df_holo[df_holo['Timestamp'] > list_target_in[0]]
                dur_over = np.sum(list_duration_in)
                mean_over = np.sum(list_duration_in) / num_over
                sd_over = np.std(list_duration_in, dtype=np.float64)
                angular_diff_mean = df_trial['TargetAngularDistance'].values.mean()
                angular_diff_sd = df_trial['TargetAngularDistance'].values.std()
                angular_max = df_trial['TargetAngularDistance'].values.max()
                angular_min = df_trial['TargetAngularDistance'].values.min()
                head_rotation_x_max = df_trial['HeadRotationX'].values.max()
                head_rotation_y_max = df_trial['HeadRotationY'].values.max()
                head_rotation_z_max = df_trial['HeadRotationZ'].values.max()
                head_rotation_x_min = df_trial['HeadRotationX'].values.min()
                head_rotation_y_min = df_trial['HeadRotationY'].values.min()
                head_rotation_z_min = df_trial['HeadRotationZ'].values.min()
                head_rotation_x_range = head_rotation_x_max - head_rotation_x_min
                head_rotation_y_range = head_rotation_y_max - head_rotation_y_min
                head_rotation_z_range = head_rotation_z_max - head_rotation_z_min
                outputList.append(
                    [subject, *current_info, total_time, frame_rate, start_time, num_over, dur_over, mean_over, sd_over,
                     angular_diff_mean, angular_diff_sd, angular_max, angular_min, head_rotation_x_max,
                     head_rotation_y_max, head_rotation_z_max, head_rotation_x_min, head_rotation_y_min,
                     head_rotation_z_min, head_rotation_x_range, head_rotation_y_range, head_rotation_z_range])
                print('-' * 50)


            except ValueError as err:
                print(subject, current_info, err)
    output = pd.DataFrame(outputList,
                          columns=['subject', 'target', 'env', 'pos', 'block', 'total_time', 'frame_rate', 'start_time',
                                   'num_over', 'dur_over', 'mean_over', 'sd_over', 'angular_diff_mean',
                                   'angular_diff_sd', 'angular_max', 'angular_min', 'head_rotation_x_max',
                                   'head_rotation_y_max', 'head_rotation_z_max', 'head_rotation_x_min',
                                   'head_rotation_y_min',
                                   'head_rotation_z_min', 'head_rotation_x_range', 'head_rotation_y_range',
                                   'head_rotation_z_range'])
    output.to_csv(path_or_buf=(DATA_ROOT / 'hololens_analysis' / '1st_analysis.csv'), index=False)


def check_files():
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                imu_data = get_file_by_info(imu_file_list, current_info)
                eye_data = get_file_by_info(eye_file_list, current_info)
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_imu = manipulate_imu(file_as_pandas(imu_data))
                # df_eye = manipulate_eye(file_as_pandas(eye_data))
                df_eye = file_as_pandas(eye_data, refined=True)
                df_holo = file_as_pandas(hololens_data)
                check_file(df_imu, df_eye, df_holo, current_info)

            except ValueError as err:
                print(subject, current_info, err)


def summary_eye_data():
    # ['circle_3d', 'confidence', 'timestamp', 'diameter_3d', 'ellipse',
    #        'location', 'diameter', 'sphere', 'projected_sphere',
    #        'model_confidence', 'model_id', 'model_birth_timestamp', 'theta', 'phi',
    #        'norm_pos', 'topic', 'id', 'method']
    output_list = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye_data = get_file_by_info(eye_file_list, current_info)
                df_eye = file_as_pandas(eye_data, refined=True)
                check_refined_eye_file(df_eye, current_info)
                _start_time = df_eye.head(1)['timestamp'].values[0]
                df_trial = df_eye[df_eye['timestamp'] > (_start_time + 1.5)]

                mean_confidence = df_trial['confidence'].mean()
                mean_theta = df_trial['theta'].mean()
                max_theta = df_trial['theta'].max()
                min_theta = df_trial['theta'].min()
                range_theta = max_theta - min_theta
                mean_phi = df_trial['phi'].mean()
                max_phi = df_trial['phi'].max()
                min_phi = df_trial['phi'].min()
                range_phi = max_phi - min_phi
                df_norm_pos = df_trial['norm_pos']
                norm_pos_x = []
                norm_pos_y = []
                for row in df_norm_pos:
                    output = row.split('[')[1]
                    output = output.split(']')[0]
                    output = output.split(',')
                    row_x = output[0]
                    row_y = output[1]
                    if 'Decimal' in output[0]:
                        row_x = output[0].split('(')[1]
                        row_x = row_x.split(')')[0]
                        row_x = ''.join(c for c in row_x if c.isdigit() or c == '.')

                    if 'Decimal' in output[1]:
                        row_y = output[1].split('(')[1]
                        row_y = row_y.split(')')[0]
                        row_y = ''.join(c for c in row_y if c.isdigit() or c == '.')
                    norm_pos_x.append(float(row_x))
                    norm_pos_y.append(float(row_y))
                mean_norm_pos_x = statistics.mean(norm_pos_x)
                max_norm_pos_x = max(norm_pos_x)
                min_norm_pos_x = min(norm_pos_x)
                range_norm_pos_x = max_norm_pos_x - min_norm_pos_x
                mean_norm_pos_y = statistics.mean(norm_pos_y)
                max_norm_pos_y = max(norm_pos_y)
                min_norm_pos_y = min(norm_pos_y)
                range_norm_pos_y = max_norm_pos_y - min_norm_pos_y

                output_list.append(
                    [subject, *current_info, mean_confidence, mean_norm_pos_x, max_norm_pos_x, min_norm_pos_x,
                     range_norm_pos_x, mean_norm_pos_y, max_norm_pos_y, min_norm_pos_y, range_norm_pos_y, mean_phi,
                     max_phi, min_phi, range_phi, mean_theta, max_theta, min_theta,
                     range_theta])
            except ValueError as err:
                print(subject, current_info, err)
    df_output = pd.DataFrame(output_list, columns=[
        'subject', 'target', 'env', 'pos', 'block', 'mean_confidence', 'mean_norm_pos_x', 'max_norm_pos_x',
        'min_norm_pos_x',
        'range_norm_pos_x', 'mean_norm_pos_y', 'max_norm_pos_y', 'min_norm_pos_y', 'range_norm_pos_y',
        'mean_phi', 'max_phi', 'min_phi', 'range_phi', 'mean_theta', 'max_theta', 'min_theta',
        'range_theta'
    ])
    df_output.to_csv(path_or_buf=(DATA_ROOT / 'refined_eye_data_analysis' / '1st_analysis.csv'), index=False)


def manual_filter():
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:

                imu_data = get_file_by_info(imu_file_list, current_info)
                eye_data = get_file_by_info(eye_file_list, current_info)
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_imu = manipulate_imu(file_as_pandas(imu_data))
                df_eye = file_as_pandas(eye_data, refined=True)
                df_holo = file_as_pandas(hololens_data)
                check_imu_file(df_imu, current_info)
                check_refined_eye_file(df_eye, current_info)
                check_holo_file(df_holo, current_info)

                timestamp_imu = df_imu['IMUtimestamp'] - df_imu['IMUtimestamp'][0]
                timestamp_holo = df_holo['Timestamp'] - df_holo['Timestamp'][0]
                timestamp_eye = df_eye['timestamp'] - df_eye['timestamp'][0]

                # df_eye = df_eye[df_eye['confidence']>0.6]

                df_eye_phi = pd.Series(list(map(float, df_eye['phi'])))
                df_eye_the = pd.Series(list(map(float, df_eye['theta'])))

                df_norm_pos = df_eye['norm_pos']
                norm_pos_x = []
                norm_pos_y = []
                for row in df_norm_pos:
                    output = row.split('[')[1]
                    output = output.split(']')[0]
                    output = output.split(',')
                    row_x = output[0]
                    row_y = output[1]
                    if 'Decimal' in output[0]:
                        row_x = output[0].split('(')[1]
                        row_x = row_x.split(')')[0]
                        row_x = ''.join(c for c in row_x if c.isdigit() or c == '.')

                    if 'Decimal' in output[1]:
                        row_y = output[1].split('(')[1]
                        row_y = row_y.split(')')[0]
                        row_y = ''.join(c for c in row_y if c.isdigit() or c == '.')
                    norm_pos_x.append(float(row_x))
                    norm_pos_y.append(float(row_y))
                df_eye_x = pd.Series(norm_pos_x)
                df_eye_y = pd.Series(norm_pos_y)

                filtered_phi, filtered_the = eye_one_euro_filtering(df_eye_phi, df_eye_the)
                filtered_x, filtered_y = eye_one_euro_filtering(df_eye_x, df_eye_y)

                interpolation_imu_z = interpolate.interp1d(timestamp_imu, df_imu['rotationZ'])
                interpolation_imu_y = interpolate.interp1d(timestamp_imu, df_imu['rotationY'])
                interpolation_imu_x = interpolate.interp1d(timestamp_imu, df_imu['rotationX'])
                interpolation_eye_phi = interpolate.interp1d(timestamp_eye, filtered_phi)
                interpolation_eye_the = interpolate.interp1d(timestamp_eye, filtered_the)
                interpolation_eye_x = interpolate.interp1d(timestamp_eye, filtered_x)
                interpolation_eye_y = interpolate.interp1d(timestamp_eye, filtered_y)
                interpolation_eye_phi_raw = interpolate.interp1d(timestamp_eye, df_eye_phi)
                interpolation_eye_the_raw = interpolate.interp1d(timestamp_eye, df_eye_the)
                interpolation_eye_x_raw = interpolate.interp1d(timestamp_eye, df_eye_x)
                interpolation_eye_y_raw = interpolate.interp1d(timestamp_eye, df_eye_y)

                interpolate_normalized_imu_Z = interpolate.interp1d(timestamp_imu, df_imu['rotationZ'])
                interpolate_normalized_pupil_Y = interpolate.interp1d(timestamp_eye, filtered_x)
                ximu = np.arange(2, 6.4, 0.005)
                xeye = np.arange(2, 6.4, 0.005)

                imuZ = interpolation_imu_z(ximu)
                imuY = interpolation_imu_y(ximu)
                imuX = interpolation_imu_x(ximu)

                eyeX_raw = interpolation_eye_x_raw(xeye)
                eyeY_raw = interpolation_eye_y_raw(xeye)
                eyePhi_raw = interpolation_eye_phi_raw(xeye)
                eyeThe_raw = interpolation_eye_the_raw(xeye)
                eyeX = interpolation_eye_x(xeye)
                eyeY = interpolation_eye_y(xeye)
                eyePhi = interpolation_eye_phi(xeye)
                eyeThe = interpolation_eye_the(xeye)

                # sd = signal_detection.real_time_peak_detection(eyeX_raw[0:2],lag=1,threshold=10,influence=0.5)
                # for row in eyeX_raw:
                #     sd.thresholding_algo(row)

                fig = plt.figure(figsize=(6, 6))
                # fig.title(str(subject) + str(current_info))
                ax1 = plt.subplot(4, 1, 1)
                ax2 = plt.subplot(4, 1, 2)
                ax3 = plt.subplot(4, 1, 3)
                ax4 = plt.subplot(4, 1, 4)

                peaks, _ = find_peaks(eyeX_raw, height=eyeX_raw.mean() + 0.02)

                # ax1.plot(eyeX)

                # ax2.plot(df_eye['confidence'],'r-')
                # ax2.hlines(0.6,xmin=10,xmax=500)
                # ax1.hlines(df_eye_x.mean(),xmin=1,xmax=500,color='r')
                # ax2.plot(eyeX_raw)
                # ax2.plot(peaks,eyeX_raw[peaks],'x')
                ax1.plot(imuZ)
                ax2.plot(imuX)
                ax3.plot(eyePhi_raw)
                ax4.plot(eyeThe_raw)

                plt.show()
            except ValueError as err:
                print(subject, current_info, err)


def main():
    use_refined_data = True
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(202, refined=use_refined_data)
    current_info = [0, 'W', 'W', 2]
    imu_data = get_file_by_info(imu_file_list, current_info)
    eye_data = get_file_by_info(eye_file_list, current_info)
    hololens_data = get_file_by_info(hololens_file_list, current_info)

    df_imu = manipulate_imu(file_as_pandas(imu_data))
    df_eye = manipulate_eye(file_as_pandas(eye_data)) if use_refined_data is False else file_as_pandas(eye_data,
                                                                                                       refined=use_refined_data)
    df_holo = file_as_pandas(hololens_data)

    timestamp_imu = df_imu['IMUtimestamp'] - df_imu['IMUtimestamp'][0]
    timestamp_holo = df_holo['Timestamp'] - df_holo['Timestamp'][0]
    timestamp_eye = df_eye['timestamp'] - df_eye['timestamp'][0]

    df_eye_phi = pd.Series(list(map(float, df_eye['phi'])))
    df_eye_the = pd.Series(list(map(float, df_eye['theta'])))

    filtered_phi, filtered_the = eye_one_euro_filtering(df_eye_phi, df_eye_the)

    interpolate_normalized_imu_Z = interpolate.interp1d(timestamp_imu, df_imu['rotationZ'])

    # ax1.plot(df_eye['phi'])
    # ax1.plot(filtered_x)
    # ax3.plot(-1*df_imu['rotationZ'])
    ax2.plot(-1 * df_imu['rotationZ'])
    ax3.plot(df_holo['HeadRotationX'])
    plt.show()


if __name__ == "__main__":
    create_eye_csv()
# from FileHandling import *
# from DataManipulation import*
# import matplotlib.pyplot as plt
# from scipy import interpolate
# # import scipy.signal
# import numpy as np
# import pandas as pd
# from OneEuroFilter_original import *
#
# from SecondAnalysis.DataManipulation import manipulate_imu, manipulate_eye
# from SecondAnalysis.FileHandling import file_as_pandas, get_file_by_info
# from SecondAnalysis.OneEuroFilter_original import config, OneEuroFilter
#
# subjects = range(201, 216)
# envs = ['U', 'W']
# poss = ['S', 'W']
# blocks = range(5)
#
#
# def normalize(dataset):
#     dataset = ((dataset - dataset.mean()) / (dataset.max() - dataset.min()))
#     return dataset
#
#
# def oneeurofilter(phi, the):
#     filtered_x = []
#     filtered_y = []
#     oneeurox = OneEuroFilter(**config)
#     oneeuroy = OneEuroFilter(**config)
#
#     for i in range(len(phi)):
#         filtered_x.append(oneeurox(phi[i]))
#         filtered_y.append(oneeuroy(the[i]))
#
#     return filtered_x, filtered_y
#
#
# def main():
#     # fig, ((ax0,ax4), (ax1,ax5) ,(ax2,ax6), (ax3,ax7), (ax8,ax9), (ax10,ax11) ) = plt.subplots(6,2, sharex=True, sharey=True)
#     fig, ((ax8,ax10,ax9,ax11)) = plt.subplots(4,1, sharex=True, sharey=True)
#     [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(202)
#
#     current_info = [0, 'U', 'W', 4]
#
#     imu_data = get_file_by_info(imu_file_list, current_info)
#     eye_data = get_file_by_info(eye_file_list, current_info)
#     hololens_data = get_file_by_info(hololens_file_list, current_info)
#     # check files
#     # Hololens: count, distance check,
#     # IMU: count,
#     # pupil: count, confidence,
#
#     df_imu = manipulate_imu(file_as_pandas(imu_data))
#     df_holo = file_as_pandas(hololens_data)
#     df_eye = manipulate_eye(file_as_pandas(eye_data))
#
#     timestamp_imu = df_imu['IMUtimestamp'] - df_imu['IMUtimestamp'][0]
#     timestamp_holo = df_holo['Timestamp'] - df_holo['Timestamp'][0]
#     timestamp_eye = df_eye['timestamp'] - df_eye['timestamp'][0]
#
#     ax8.set_xlim([2, 5.5])
#     ax8.set_ylim([-0.5, 0.5])
#
#     df_eye_phi = pd.Series(list(map(float, df_eye['phi'])))
#     df_eye_the = pd.Series(list(map(float, df_eye['theta'])))
#
#     filtered_x, filtered_y = oneeurofilter(df_eye_phi, df_eye_the)
#     filtered_x = pd.Series(filtered_x)
#     filtered_y = pd.Series(filtered_y)
#
#     # Normalization vertical
#     df_imu['rotationZ'] = normalize(df_imu['rotationZ'])
#     df_holo['HeadRotationY'] = normalize(df_holo['HeadRotationY'])
#     df_eye_phi = normalize(df_eye_phi)
#     filtered_x = normalize(filtered_x)
#     # Normalization horizontal
#     df_imu['rotationX'] = normalize(df_imu['rotationX'])
#     df_holo['HeadRotationX'] = normalize(df_holo['HeadRotationX'])
#     df_eye_the = normalize(df_eye_the)
#     filtered_y = normalize(filtered_y)
#     # Normalization others
#     df_imu['rotationY'] = normalize(df_imu['rotationY'])
#     df_holo['HeadRotationZ'] = normalize(df_holo['HeadRotationZ'])
#
#     # # Plotting
#     # ax0.plot(timestamp_imu, -df_imu['rotationZ'])
#     # ax1.plot(timestamp_holo, df_holo['HeadRotationY'])
#     # ax2.plot(timestamp_eye, df_eye_phi)
#     # ax3.plot(timestamp_eye, filtered_x)
#     # # ax3.plot(TimeStamp_imu, -df_imu['rotationZ'], 'darkred', TimeStamp_eye, df_eye_phi, 'darkblue')
#     # ax4.plot(timestamp_imu, -df_imu['rotationX'])
#     # ax5.plot(timestamp_holo, df_holo['HeadRotationX'])
#     # ax6.plot(timestamp_eye, -df_eye_the)
#     # ax7.plot(timestamp_eye, -filtered_y)
#     # ax7.plot(TimeStamp_imu, -df_imu['rotationX'], 'darkred', TimeStamp_eye, -df_eye_the, 'darkblue')
#     # ax8.plot(timestamp_imu, -df_imu['rotationZ'], 'darkred', timestamp_eye-0.05, filtered_x, 'darkblue')
#     # ax9.plot(timestamp_imu, -df_imu['rotationX'], 'darkred', timestamp_eye-0.1, -filtered_y, 'darkblue')
#     # ax8.plot(TimeStamp_imu, -df_imu['rotationY'])
#     # ax9.plot(TimeStamp_holo, df_holo['HeadRotationZ'])
#
#     interpolate_normalized_imu_Z = interpolate.interp1d(timestamp_imu, df_imu['rotationZ'])
#     interpolate_normalized_pupil_Y = interpolate.interp1d(timestamp_eye, filtered_x)
#     ximu = np.arange(2, 6.4, 0.005)
#     xeye = np.arange(2, 6.4, 0.005)
#     imuZ = interpolate_normalized_imu_Z(ximu-0.05)
#     eyeY = interpolate_normalized_pupil_Y(xeye)
#     imuZ = normalize(imuZ)
#     eyeY = normalize(eyeY)
#     ax8.plot(ximu, imuZ, 'darkred', xeye, -eyeY, 'darkblue')
#     ax10.plot(ximu, (imuZ-eyeY)/2)
#
#     interpolate_normalized_imu_X = interpolate.interp1d(timestamp_imu, df_imu['rotationX'])
#     interpolate_normalized_pupil_X = interpolate.interp1d(timestamp_eye, filtered_y)
#     ximu = np.arange(2, 6.4, 0.005)
#     xeye = np.arange(2, 6.4, 0.005)
#     imuX = interpolate_normalized_imu_X(ximu-0.1)
#     eyeX = interpolate_normalized_pupil_X(xeye)
#     imuX = normalize(imuX)
#     eyeX = normalize(eyeX)
#     ax9.plot(ximu, -imuX, 'darkred', xeye, -eyeX, 'darkblue')
#     ax11.plot(ximu, (-imuX-eyeX)/2)
#     # ax11.plot(xnew, yeye, '-')
#     # print(interpolate_normalized_imu_X)
#     # df_eye_phi = pd.Series(list(map(float, interpolate_normalized_imu_X)))
#
#
#     # a10.plot(interpolate_normalized_imu_X)
#     # a11.plot(interpolate_normalized_pupil_Y)
#
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
