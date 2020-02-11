import pandas as pd
from QuaternionHandling import *
import demjson

def manipulate_imu(_imu_dataframe: pd.DataFrame):
    angle_list = []
    for row in _imu_dataframe.itertuples(index=False):
        euler_angle = quaternion_to_euler(row[1], row[2], row[3], row[4])
        euler_angle = (row[0],) + euler_angle
        angle_list.append(euler_angle)
    output = pd.DataFrame(data=angle_list, columns=['IMUtimestamp', 'rotationX', 'rotationY', 'rotationZ'])
    return output

def change_eye_dataframe(_eye_dataframe: pd.DataFrame):
    eye_list = []
    for row in _eye_dataframe.itertuples(index=False):
        python_timestamp = row[0]
        pupil_data = row[1]
        json_dict = demjson.decode(pupil_data)
        eye_list.append(json_dict)
    output = pd.DataFrame(eye_list)
    print(output)
    # for row in _dataframe.itertuples():
    #     python_timestamp = row[1]
    #     pupil_row = row[2]
    #     json_dict = demjson.decode(pupil_row)
    #     print(json_dict['timestamp'], json_dict['norm_pos'], json_dict['phi'], json_dict['theta'])
    #     output.append({'python_timestamp': python_timestamp, 'timestamp': json_dict['timestamp'],
    #                    'norm_pos_x': json_dict['norm_pos'][0], 'norm_pos_y': json_dict['norm_pos'][1],
    #                    'phi': json_dict['phi'], 'theta': json_dict['theta']}, ignore_index=True)
    # _dataframe.drop(_dataframe.columns[2:], axis=1, inplace=True)
    # _dataframe.columns = ['python_timestamp', 'pupil_data']
    # return _dataframe