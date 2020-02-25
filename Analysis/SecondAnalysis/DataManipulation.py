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


def manipulate_eye(_eye_dataframe: pd.DataFrame):
    eye_list = []
    for row in _eye_dataframe.itertuples(index=False):
        python_timestamp = row[0]
        pupil_data = row[1]
        json_dict = demjson.decode(pupil_data)
        eye_list.append(json_dict)
    output = pd.DataFrame(eye_list)
    output = output[output.confidence > 0.6]
    return output


def check_files(_imu_dataframe: pd.DataFrame, _eye_dataframe: pd.DataFrame, _holo_dataframe: pd.DataFrame, trial_info):
    if _imu_dataframe is None or _eye_dataframe is None or _holo_dataframe is None:
        return ValueError('there is empty file!')
    if _imu_dataframe.shape[0] < 600:
        return ValueError('too short IMU data')
    if _eye_dataframe.shape[0] < 600:
        return ValueError('too short EYE data')
    if _holo_dataframe.shape[0] < 300:
        return ValueError('too short HOLO data')
    head_position_start = _holo_dataframe.head(1)['HeadPositionZ']
    head_position_end = _holo_dataframe.tail(1)['HeadPositionZ']
    if head_position_end - head_position_start < 5 and trial_info[2] == 'W':
        return ValueError('Short walk length:', (head_position_end - head_position_start))
