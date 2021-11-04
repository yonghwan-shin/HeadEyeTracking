import pandas as pd
from pathlib import Path
import json
import math


def get_one_trial(subject, posture, cursor_type, repetition, end_num):
    data = read_hololens_data(subject, posture, cursor_type, repetition)
    splited_data = split_target(data)
    temp_data = splited_data[end_num]
    temp_data.reset_index(inplace=True)
    temp_data.timestamp -= temp_data.timestamp.values[0]
    initial_contact_time = temp_data[temp_data.target_name == "Target_" + str(end_num)].timestamp.values[
        0]
    dwell_temp = temp_data[temp_data.timestamp > initial_contact_time]
    dwell_temp['cursor_rotation'] = dwell_temp.apply(
        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
    dwell_temp['target_rotation'] = dwell_temp.apply(
        lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                              x.target_position_z - x.origin_z), axis=1)
    dwell_temp['offset_horizontal'] = dwell_temp.apply(
        lambda x: x.cursor_rotation[1] - x.target_rotation[1], axis=1)
    dwell_temp['offset_vertical'] = dwell_temp.apply(
        lambda x: x.cursor_rotation[0] - x.target_rotation[0], axis=1)
    return dwell_temp


def read_hololens_data(subject, posture, cursor_type, repetition):
    root = Path(__file__).resolve().parent / 'data' / str(subject)
    trial_detail = f'subject{str(subject)}_posture{str(posture)}_cursor{str(cursor_type)}_repetition{str(repetition)}'
    files = root.rglob(trial_detail + '*.json')
    try:
        for file in files:
            if trial_detail in file.name:
                with open(file) as f:  # found exact file
                    # output = pd.DataFrame(json.load(f))
                    output = pd.read_json(f)
                    target_position = pd.json_normalize(output.target_position, sep='_').rename(
                        columns={'x': 'target_position_x', 'y': 'target_position_y', 'z': 'target_position_z'})
                    head = pd.json_normalize(output.headData, sep='_')
                    cursor = pd.json_normalize(output.cursorData, sep='_')
                    output.drop(['target_position', 'headData', 'cursorData'], axis='columns', inplace=True)
                    output = pd.concat([output, target_position, head, cursor], axis=1)
                    return output
    except Exception as e:
        print(e.args)


def split_target(data):
    output = []
    data = data[data['step_num'] != 0]
    # first_end_num = data['end_num'].values[0]
    # for i in range(len(data) - 1):
    #     if data['end_num'].values[i] == first_end_num:
    #         pass
    #         # data = data.drop(i)
    #         # print(first_end_num, 'drop', i, len(data))
    #     else:
    #         # print(i, data['timestamp'].values[i],data['end_num'].values[i],first_end_num)
    #         data=data.drop([x for x in range(i)])
    #         break
    for target_num in range(9):
        output.append(data[data['end_num'] == target_num])
    return output

#
# def change_angle(a):
#     if abs(a)>
#     return a


def asSpherical(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return [0, 0]
    theta = math.acos(z / r) * 180 / math.pi
    phi = math.atan2(y, x) * 180 / math.pi
    return [theta, phi]
