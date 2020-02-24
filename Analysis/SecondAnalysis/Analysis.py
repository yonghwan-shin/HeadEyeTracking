from FileHandling import *
from DataManipulation import*
import matplotlib.pyplot as plt

subjects = range(201, 216)
envs = ['U', 'W']
poss = ['S', 'W']
blocks = range(5)


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
    ax2.plot(df_holo['Timestamp'],df_holo['HeadRotationX'])
    # plt.plot(df_holo['HeadRotationX'])
    # plt.plot(eye_refined_data)

    plt.show()


if __name__ == "__main__":
    main()
