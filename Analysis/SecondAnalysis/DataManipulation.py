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
    print(output)
    return output
