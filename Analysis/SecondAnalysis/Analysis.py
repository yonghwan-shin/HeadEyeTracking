from FileHandling import *


subjects = range(1, 16)
envs = ['U', 'W']
poss = ['S', 'W']
blocks = range(5)



def main():
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(6)
    current_info = [1, 'U', 'S', 1]
    imu_data = get_file_by_info(imu_file_list,current_info)
    eye_data = get_file_by_info(eye_file_list,current_info)
    hololens_data = get_file_by_info(hololens_file_list, current_info)
    df_imu = file_as_pandas(imu_data)
    df_eye = change_eye_dataframe(file_as_pandas(eye_data))
    df_holo = file_as_pandas(hololens_data)




if __name__ == "__main__":
    main()
