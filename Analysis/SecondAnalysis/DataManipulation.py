import pandas as pd
from QuaternionHandling import *
import demjson
from scipy import interpolate
import numpy as np


def maprange(a, b, s):
    (a1, a2), (b1, b2) = a, b
    return b1 + ((s - a1) * (b2 - b1) / (a2 - a1))


def centralise_dataframes(_df, *args):
    for column in args:
        _df[column] = _df[column] - _df[column].head(1).values[0]
    return _df


def make_x_axis(timestamp: pd.Series, interval: float):
    # print(timestamp.tail(1).values[0] - timestamp.head(1).values[0])
    if timestamp.tail(1).values[0] - timestamp.head(1).values[0] > 2:
        return np.arange(timestamp.head(1).values[0]+1, timestamp.tail(1).values[0], interval)
    else:
        raise ValueError('too short time in eye')


def manipulate_imu(_imu_dataframe: pd.DataFrame):
    angle_list = []
    try:
        for row in _imu_dataframe.itertuples(index=False):
            euler_angle = quaternion_to_euler(row[1], row[2], row[3], row[4])
            x = -euler_angle[0] if (-euler_angle[0] < 180) else -euler_angle[0] - 360
            y = -euler_angle[1] if (-euler_angle[1] < 180) else -euler_angle[1] - 360
            z = -euler_angle[2] - 180 if (-euler_angle[2] - 180 > -180) else -euler_angle[2] + 180

            euler_angle = (x, y, z)
            output = (row[0],) + euler_angle

            angle_list.append(output)
        output = pd.DataFrame(data=angle_list, columns=['IMUtimestamp', 'rotationX', 'rotationY', 'rotationZ'])
        return output
    except:
        raise ValueError("fail in IMU manipulation")


def manipulate_eye(_eye_dataframe: pd.DataFrame):
    try:
        eye_list = []
        for row in _eye_dataframe.itertuples(index=False):
            python_timestamp = row[0]
            pupil_data = row[1]
            json_dict = demjson.decode(pupil_data)
            json_dict['python_timestamp'] = python_timestamp
            eye_list.append(json_dict)
        output = pd.DataFrame(eye_list)
        # output = output[output['confidence'] > 0.6]
        return output
    except:
        raise ValueError('fail in EYE manipulation')
        # print('fail in eye manipulation')


def check_imu_file(_imu_dataframe: pd.DataFrame, trial_info):
    if _imu_dataframe is None:
        raise ValueError('none imu data')
    if _imu_dataframe.shape[0] < 600:
        raise ValueError('too short IMU data')
    return None


def check_holo_file(_holo_dataframe: pd.DataFrame, trial_info):
    if _holo_dataframe is None:
        raise ValueError('none holo data')
    if _holo_dataframe.shape[0] < 300:
        raise ValueError('too short HOLO data', _holo_dataframe.shape[0])
    head_position_start = float(_holo_dataframe.head(1)['HeadPositionZ'])
    head_position_end = float(_holo_dataframe.tail(1)['HeadPositionZ'])
    if (head_position_end - head_position_start < 4.5) & (trial_info[2] == 'W'):
        raise ValueError('Short walk length:', (head_position_end - head_position_start))
    # print('no errors in this files')
    return None


def check_refined_eye_file(_eye_dataframe: pd.DataFrame, trial_info):
    if _eye_dataframe is None:
        raise ValueError('none eye data')
    if _eye_dataframe.shape[0] < 600:
        raise ValueError('too short EYE data', _eye_dataframe.shape[0])
    return None


def check_file(_imu_dataframe: pd.DataFrame, _eye_dataframe: pd.DataFrame, _holo_dataframe: pd.DataFrame, trial_info):
    # print('checking',trial_info, _imu_dataframe.shape[0],_eye_dataframe.shape[0],_holo_dataframe.shape[0])
    # if trial_info[2] =='W':
    #     head_position_start = float(_holo_dataframe.head(1)['HeadPositionZ'])
    #     head_position_end = float(_holo_dataframe.tail(1)['HeadPositionZ'])
    #     print((head_position_end - head_position_start))
    if _imu_dataframe is None or _eye_dataframe is None or _holo_dataframe is None:
        raise ValueError('there is empty file!', )
    if _imu_dataframe.shape[0] < 600:
        raise ValueError('too short IMU data', _imu_dataframe.shape[0])
    if _eye_dataframe.shape[0] < 600:
        raise ValueError('too short EYE data', _eye_dataframe.shape[0])
    if _holo_dataframe.shape[0] < 300:
        raise ValueError('too short HOLO data', _holo_dataframe.shape[0])
    head_position_start = float(_holo_dataframe.head(1)['HeadPositionZ'])
    head_position_end = float(_holo_dataframe.tail(1)['HeadPositionZ'])
    if (head_position_end - head_position_start < 4.5) & (trial_info[2] == 'W'):
        raise ValueError('Short walk length:', (head_position_end - head_position_start))
    # print('no errors in this files')
    return None


def get_timestamp(dataframe: pd.DataFrame, column_name: str):
    return dataframe[column_name] - dataframe[column_name][0]


def get_interpolation_function(timestamp, *args):
    output = []
    for data in args:
        output.append(interpolate.interp1d(timestamp, data))
    return output
# Raw Pupil data norm-pos manipulation code,, not used now
# for row in df_norm_pos:
#     output = row.split('[')[1]
#     output = output.split(']')[0]
#     output = output.split(',')
#     row_x = output[0]
#     row_y = output[1]
#     if 'Decimal' in output[0]:
#         row_x = output[0].split('(')[1]
#         row_x = row_x.split(')')[0]
#         row_x = ''.join(c for c in row_x if c.isdigit() or c == '.')
#
#     if 'Decimal' in output[1]:
#         row_y = output[1].split('(')[1]
#         row_y = row_y.split(')')[0]
#         row_y = ''.join(c for c in row_y if c.isdigit() or c == '.')
#     norm_pos_x.append(float(row_x))
#     norm_pos_y.append(float(row_y))
