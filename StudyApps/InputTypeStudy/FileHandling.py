import itertools

import pandas as pd
from pathlib import Path
import json
import math
import numpy as np
import numpy.linalg as LA
import os

default_target_size = 3.0 / 2
default_target_radius = 0.05237
default_target_diameter = 0.10474

sigmas = {('EYE', 'WALK', 'horizontal'): 4.420237751534142,
          ('EYE', 'WALK', 'vertical'): 2.4375580926867078,
          ('EYE', 'STAND', 'horizontal'): 1.5635038623192548,
          ('EYE', 'STAND', 'vertical'): 1.491778058469321,
          ('HAND', 'WALK', 'horizontal'): 6.521336309396893,
          ('HAND', 'WALK', 'vertical'): 1.6178699940290733,
          ('HAND', 'STAND', 'horizontal'): 1.2868251691549768,
          ('HAND', 'STAND', 'vertical'): 1.3437840646867873,
          ('HEAD', 'WALK', 'horizontal'): 5.0511439371221885,
          ('HEAD', 'WALK', 'vertical'): 2.3182985184738376,
          ('HEAD', 'STAND', 'horizontal'): 1.303755389483091,
          ('HEAD', 'STAND', 'vertical'): 1.5906082672928836}

# wide = 1
# x_offsets = [wide * math.sin(t * math.pi / 9 * 2) for t in range(9)]
# y_offsets = [wide * math.cos(t * math.pi / 9 * 2) for t in range(9)]
# for i in range(9):
#     j = i + 5
#     if j > 8:
#         j -= 9
#     x_dir = x_offsets[i] - x_offsets[j]
#     y_dir = y_offsets[i] - y_offsets[j]
#     print(i, ':',(x_dir, y_dir),',')
directions = {
    0: (0.34202014332566866, 1.9396926207859084),
    1: (1.5088130134709776, 1.2660444431189786),
    2: (1.9696155060244163, 4.440892098500626e-16),
    3: (1.5088130134709785, -1.2660444431189777),
    4: (0.3420201433256689, -1.9396926207859084),
    5: (-0.9848077530122079, -1.7057370639048866),
    6: (-1.8508331567966465, -0.6736481776669309),
    7: (-1.850833156796647, 0.6736481776669297),
    8: (-0.9848077530122085, 1.7057370639048863),
}


def get_one_trial(subject, posture, cursor_type, repetition, end_num):
    data = read_hololens_data(subject, posture, cursor_type, repetition)
    splited_data = split_target(data)
    temp_data = splited_data[end_num]
    temp_data.reset_index(inplace=True)
    temp_data.timestamp -= temp_data.timestamp.values[0]
    # initial_contact_time = temp_data[temp_data.target_name == "Target_" + str(end_num)].timestamp.values[
    #     0]
    # dwell_temp = temp_data[temp_data.timestamp > initial_contact_time]
    # dwell_temp['cursor_rotation'] = dwell_temp.apply(
    #     lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
    # dwell_temp['target_rotation'] = dwell_temp.apply(
    #     lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
    #                           x.target_position_z - x.origin_z), axis=1)
    # dwell_temp['offset_horizontal'] = dwell_temp.apply(
    #     lambda x: x.cursor_rotation[1] - x.target_rotation[1], axis=1)
    # dwell_temp['offset_vertical'] = dwell_temp.apply(
    #     lambda x: x.cursor_rotation[0] - x.target_rotation[0], axis=1)
    return temp_data


def validate_trial_data(data, cursor_type):
    # if there is a sudden target-shift occurs
    if cursor_type == 'EYE':
        # drop_index = temp_data[(temp_data['ray_direction_x'] == 0) & (temp_data['ray_direction_y'] == 0) & (
        #         temp_data['ray_direction_z'] == 0)].index
        data['check_eye'] = data.latestEyeGazeDirection_x.diff(1)
        eye_index = data[data.check_eye == 0].index
        invalidate_index = data[data.isEyeTrackingEnabledAndValid == False].index
        loss_interval = 3
        loss_indices = []
        for i in range(-loss_interval, loss_interval + 1):
            loss_indices += list(eye_index + i)
        for i in loss_indices:
            if i > len(data) or i < 0:
                loss_indices.remove(i)
        # for i in range(-loss_interval, loss_interval + 1):
        #     if len(data) + i in loss_indices:
        #         loss_indices.remove(len(data) + i)
        if len(eye_index) > len(data.index) / 3:
            # if len(drop_index) > 0 and len(drop_index) > 0:
            # print('eye loss')
            return False, 'loss'
    else:
        drop_index = data[(data['direction_x'] == 0) & (data['direction_y'] == 0) & (
                data['direction_z'] == 0)].index
        if len(drop_index) > 0 and len(drop_index) > len(data.index) / 3:
            # if len(drop_index) > 0 and len(drop_index) > 0:
            return False, 'loss'
    if len(data[data['error_frame'] == True]) > len(data.index) * 2 / 3:
        # print(len(data['error_frame']) , len(data.index))
        return False, 'LOSS'
    outlier = list(data[(abs(data.target_horizontal_velocity) > 10 * 57.296)].index)
    outlier = [x for x in outlier if x > 5]
    if len(outlier) > 1:
        # in this data, sudden target movement happened.
        return False, 'jump'
    return True, 'None'
    # print('drop length', len(drop_index), sub_num, pos, cursor_type, rep, t)


def without_cursor_file(subject, posture, cursor_type, repetition):
    root = Path(__file__).resolve().parent / 'data' / (str(subject) + '_nocursor')
    # subject = 0
    # posture = 'WALK'
    # cursor_type = 'EYE'
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
                    output['cursor_rotation'] = output.apply(
                        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                    output['target_rotation'] = output.apply(
                        lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                              x.target_position_z - x.origin_z), axis=1)

                    output['cursor_rotation'] = output.apply(
                        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                    output['target_rotation'] = output.apply(
                        lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                              x.target_position_z - x.origin_z), axis=1)
                    output['cursor_horizontal_angle'] = output.apply(
                        lambda x: x.cursor_rotation[1], axis=1
                    )
                    output['cursor_vertical_angle'] = output.apply(
                        lambda x: x.cursor_rotation[0], axis=1
                    )
                    output['target_horizontal_angle'] = output.apply(
                        lambda x: x.target_rotation[1], axis=1
                    )
                    output['target_vertical_angle'] = output.apply(
                        lambda x: x.target_rotation[0], axis=1
                    )
                    output['horizontal_offset'] = output.apply(
                        lambda x: math.degrees(math.sin(
                            math.radians(x.target_horizontal_angle - x.cursor_horizontal_angle))), axis=1
                    )
                    output['vertical_offset'] = output.apply(
                        lambda x: math.degrees(math.sin(
                            math.radians(x.target_vertical_angle - x.cursor_vertical_angle))), axis=1
                    )
                    output['abs_horizontal_offset'] = output['horizontal_offset'].apply(abs)
                    output['abs_vertical_offset'] = output['vertical_offset'].apply(abs)
                    output['target_horizontal_velocity'] = (
                            output['target_horizontal_angle'].diff(1) / output['timestamp'].diff(1))
                    # print(str(root / (file.name.split('.')[0] + ".pkl")))
                    # output.to_pickle(path=str(root / (file.name.split('.')[0] + ".pkl")))
                    return output
    except Exception as e:
        print(e.args)
    return output


def read_hololens_data(subject, posture, cursor_type, repetition, reset=False, pilot=False):
    root = Path(__file__).resolve().parent / 'data' / str(subject)
    if pilot: root = Path(__file__).resolve().parent / 'data' / (str(subject) + '_nocursor')
    trial_detail = f'subject{str(subject)}_posture{str(posture)}_cursor{str(cursor_type)}_repetition{str(repetition)}'
    files = root.rglob(trial_detail + '*.json')
    pickled_files = root.rglob(trial_detail + "*.pkl")
    try:  # for faster data import -> make/bring compressed version
        for pickled_file in pickled_files:
            if trial_detail in pickled_file.name:
                output = pd.read_pickle(pickled_file)
                if reset:
                    os.remove(pickled_file.absolute())
                    print('remove file and re-made', pickled_file.name)
                    output = read_hololens_data(subject, posture, cursor_type, repetition)
                    # drop_index = output[
                    #     (output['abs_horizontal_offset'] > 3 * sigmas[(cursor_type, posture, 'horizontal')]) | (
                    #             output['abs_vertical_offset'] > 3 * sigmas[(cursor_type, posture, 'vertical')])]
                    # output = output.drop(drop_index)
                # print('found pickled file!')
                return output
    except Exception as e:
        print('Error in reading pickled files', e.args)

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
                    output['cursor_rotation'] = output.apply(
                        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                    output['target_rotation'] = output.apply(
                        lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                              x.target_position_z - x.origin_z), axis=1)

                    output['cursor_rotation'] = output.apply(
                        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                    output['target_rotation'] = output.apply(
                        lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                              x.target_position_z - x.origin_z), axis=1)
                    output['cursor_horizontal_angle'] = output.apply(
                        lambda x: x.cursor_rotation[1], axis=1
                    )
                    output['cursor_vertical_angle'] = output.apply(
                        lambda x: x.cursor_rotation[0], axis=1
                    )
                    output['target_horizontal_angle'] = output.apply(
                        lambda x: x.target_rotation[1], axis=1
                    )
                    output['target_vertical_angle'] = output.apply(
                        lambda x: x.target_rotation[0], axis=1
                    )

                    # print(output['target_horizontal_angle'])
                    # output['horizontal_offset'] = output.apply(
                    #     # lambda x: math.degrees(math.sin(
                    #     #     math.radians(x.target_horizontal_angle - x.cursor_horizontal_angle))), axis=1
                    #     lambda x: correct_angle(x.target_horizontal_angle - x.cursor_horizontal_angle),axis=1
                    # )

                    # output['horizontal_offset']=(output.target_horizontal_angle - output.cursor_horizontal_angle).apply(correct_angle)
                    output['horizontal_offset'] = (
                            output.target_horizontal_angle - output.cursor_horizontal_angle).apply(correct_angle)
                    output['vertical_offset'] = (
                            output.target_vertical_angle - output.cursor_vertical_angle).apply(correct_angle)
                    # output['vertical_offset'] = output.apply(
                    #     # lambda x: math.degrees(math.sin(
                    #     #     math.radians(x.target_vertical_angle - x.cursor_vertical_angle))), axis=1
                    #     lambda x: correct_angle(x.target_vertical_angle - x.cursor_vertical_angle), axis=1
                    # )
                    output['angle'] = (output.horizontal_offset ** 2 + output.vertical_offset ** 2).apply(
                        math.sqrt)
                    output['distance'] = ((output.target_position_x - output.origin_x) ** 2 + (
                            output.target_position_y - output.origin_y) ** 2 +
                                          (output.target_position_z - output.origin_z) ** 2).apply(math.sqrt)
                    output['max_angle'] = (default_target_radius / output['distance']).apply(math.asin).apply(
                        math.degrees)
                    output['success'] = output.angle < output.max_angle
                    output['abs_horizontal_offset'] = output['horizontal_offset'].apply(abs)
                    output['abs_vertical_offset'] = output['vertical_offset'].apply(abs)
                    output['target_horizontal_velocity'] = (
                            output['target_horizontal_angle'].diff(1) / output['timestamp'].diff(1))
                    print(str(root / (file.name.split('.')[0] + ".pkl")))
                    output.to_pickle(path=str(root / (file.name.split('.')[0] + ".pkl")))
                    return output
    except Exception as e:
        print('error in reading file', e.args)


def correct_angle(angle):
    if angle > 180:
        return angle - 360
    if angle < -180:
        return angle + 360
    return angle


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
    theta = math.degrees(math.acos(y / r))
    # phi  = math.degrees(math.atan(y/x))
    phi = math.degrees(math.atan2(x, z))
    return [theta, phi]


# def position_speed(x,z):


def euler(R):
    if math.isclose(R[0][2], -1.0):
        x = 0
        y = math.pi / 2
        z = x + math.atan2(R[1][0], R[2][0])
    elif math.isclose(R[0][2], 1.0):
        x = 0
        y = -math.pi / 2
        z = -x + math.atan2(-R[1][0], -R[2][0])
    else:
        x1 = -math.asin(R[0][2])
        x2 = math.pi - x1
        y1 = math.atan2(R[1][2] / math.cos(x1), R[2][2] / math.cos(x1))
        y2 = math.atan2(R[1][2] / math.cos(x2), R[2][2] / math.cos(x2))

        z1 = math.atan2(R[0][1] / math.cos(x1), R[0][0] / math.cos(x1))
        z2 = math.atan2(R[0][1] / math.cos(x2), R[0][0] / math.cos(x2))
    if abs(x1) + abs(y1) + abs(z1) <= abs(x2) + abs(y2) + abs(z2):
        return [x1, y1, z1]
    else:
        return [x2, y2, z2]


def angle_between(a, b):
    inner = np.inner(a, b)
    norms = LA.norm(a) * LA.norm(b)
    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    return deg


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def change_angle(a):
    if a < -180:
        a = a + 360
    if a > 180:
        a = a - 360
    return a
