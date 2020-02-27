from FileHandling import *
from DataManipulation import *
import matplotlib.pyplot as plt
import itertools

# subjects = range(201, 216)
subjects = range(201, 212)
targets = range(8)
envs = ['U', 'W']
poss = ['S', 'W']
blocks = range(5)


def create_eye_csv():
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye_data = get_file_by_info(eye_file_list, current_info)
                df_eye = manipulate_eye(file_as_pandas(eye_data))
                df_eye.to_csv(path_or_buf=(DATA_ROOT / 'refined_eye_data'/ ('refined_'+eye_data.name)) ,index=False)
                print('saved', eye_data.name, '  ', current_info, subject)
            except ValueError as err:
                print(err, current_info)


def check_files():
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                imu_data = get_file_by_info(imu_file_list, current_info)
                eye_data = get_file_by_info(eye_file_list, current_info)
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_imu = manipulate_imu(file_as_pandas(imu_data))
                df_eye = manipulate_eye(file_as_pandas(eye_data))
                df_holo = file_as_pandas(hololens_data)

                check_file(df_imu, df_eye, df_holo, current_info)
            except ValueError as err:
                print(err, current_info)


def main():
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(202)
    current_info = [0, 'U', 'W', 2]
    imu_data = get_file_by_info(imu_file_list, current_info)
    eye_data = get_file_by_info(eye_file_list, current_info)
    hololens_data = get_file_by_info(hololens_file_list, current_info)
    # check files
    # Hololens: count, distance check,
    # IMU: count,
    # pupil: count, confidence,

    # df_imu = file_as_pandas(imu_data)
    df_imu = manipulate_imu(file_as_pandas(imu_data))
    df_eye = manipulate_eye(file_as_pandas(eye_data))
    df_holo = file_as_pandas(hololens_data)
    # plt.plot(df_imu['IMUtimestamp'],df_imu['rotationX'])

    ax1.plot(df_eye['timestamp'], df_eye['theta'])
    # ax2.plot(df_holo['Timestamp'],df)

    # print(df_holo.columns)
    ax2.plot(df_holo['Timestamp'], df_holo['HeadRotationX'])
    # plt.plot(df_holo['HeadRotationX'])
    # plt.plot(eye_refined_data)

    plt.show()


if __name__ == "__main__":
    # main()
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


if __name__ == "__main__":
    main()
