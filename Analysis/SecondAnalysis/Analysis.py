from FileHandling import *
from DataManipulation import*
import matplotlib.pyplot as plt

subjects = range(1, 16)
envs = ['U', 'W']
poss = ['S', 'W']
blocks = range(5)


def main():
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(6)
    current_info = [0, 'U', 'W', 2]
    # imu_data = get_file_by_info(imu_file_list, current_info)
    eye_data = get_file_by_info(eye_file_list, current_info)
    hololens_data = get_file_by_info(hololens_file_list, current_info)

    # df_imu = file_as_pandas(imu_data)
    # manipulate_imu(df_imu)
    df_eye = change_eye_dataframe(file_as_pandas(eye_data))
    # eye_refined_data = []
    # for row in df_eye.itertuples():
    #     # print(row[2])
    #     eye_refined_data.append(demjson.decode(row[2])['norm_pos'][1])
    df_holo = file_as_pandas(hololens_data)



    # print(df_holo.columns)
    # plt.plot(df_holo['Timestamp'],df_holo['HeadRotationX'],'x')
    plt.plot(df_holo['HeadRotationX'])
    # plt.plot(eye_refined_data)

    plt.show()


if __name__ == "__main__":
    main()
