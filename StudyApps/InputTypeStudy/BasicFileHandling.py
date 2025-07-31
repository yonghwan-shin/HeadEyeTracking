from pathlib import Path
import json
import math
import pandas as pd

def read_data(subject=None, repetition=None, cursor=None, selection=None, posture=None, filename=None):
    root = Path(__file__).resolve().parent / 'Dataset' / str(posture) / str(subject)
    # if stand: root = Path(__file__).resolve().parent / 'data' / 'EyeTest' / 'dummy' / str(subject)
    trial_detail = f'subject{str(subject)}_cursor{str(cursor)}_Selection{str(selection)}_repetition{str(repetition)}'
    if filename != None:
        trial_detail = filename
        root = Path(__file__).resolve().parent / 'Dataset'
    files = root.rglob(trial_detail + '*.json')

    for file in files:
        if trial_detail in file.name:
            with open(file) as f:  # found exact file
                output = pd.read_json(f)
                target_position = pd.json_normalize(output.target_position, sep='_').rename(
                    columns={'x': 'target_position_x', 'y': 'target_position_y', 'z': 'target_position_z'})
                head_origin = pd.json_normalize(output.head_origin, sep='_').rename(
                    columns={'x': 'head_origin_x', 'y': 'head_origin_y', 'z': 'head_origin_z'})
                head_forward = pd.json_normalize(output.head_forward, sep='_').rename(
                    columns={'x': 'head_forward_x', 'y': 'head_forward_y', 'z': 'head_forward_z'})
                head_rotation = pd.json_normalize(output.head_rotation, sep='_').rename(
                    columns={'x': 'head_rotation_x', 'y': 'head_rotation_y', 'z': 'head_rotation_z'})
                eyeRay_direction = pd.json_normalize(output.eyeRayDirection, sep='_').rename(
                    columns={'x': 'eyeRay_direction_x', 'y': 'eyeRay_direction_y', 'z': 'eyeRay_direction_z'})
                cursor = pd.json_normalize(output.cursorData, sep='_')
                output = pd.concat([output, target_position,
                                    head_origin, head_forward, head_rotation, eyeRay_direction,
                                    cursor], axis=1)

                output['cursor_rotation'] = output.apply(
                    lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                output['target_rotation'] = output.apply(
                    lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                          x.target_position_z - x.origin_z), axis=1)
                output['head_rotation'] = output.apply(
                    lambda x: asSpherical(x.head_forward_x, x.head_forward_y, x.head_forward_z), axis=1)
                output['eyeRay_rotation'] = output.apply(
                    lambda x: asSpherical(x.eyeRay_direction_x, x.eyeRay_direction_y, x.eyeRay_direction_z), axis=1)

                output['head_horizontal_angle'] = output.apply(
                    lambda x: x.head_rotation[1], axis=1
                )
                output['head_vertical_angle'] = output.apply(
                    lambda x: x.head_rotation[0], axis=1
                )
                output['cursor_horizontal_angle'] = output.apply(
                    lambda x: x.cursor_rotation[1], axis=1
                )
                output['cursor_vertical_angle'] = output.apply(
                    lambda x: x.cursor_rotation[0], axis=1
                )
                output['eyeRay_horizontal_angle'] = output.apply(
                    lambda x: x.eyeRay_rotation[1], axis=1
                )
                output['eyeRay_vertical_angle'] = output.apply(
                    lambda x: x.eyeRay_rotation[0], axis=1
                )
                output['target_horizontal_angle'] = output.apply(
                    lambda x: x.target_rotation[1], axis=1
                )
                output['target_vertical_angle'] = output.apply(
                    lambda x: x.target_rotation[0], axis=1
                )
                output['horizontal_offset'] = (
                        output.target_horizontal_angle - output.cursor_horizontal_angle).apply(correct_angle)
                output['vertical_offset'] = (
                        output.target_vertical_angle - output.cursor_vertical_angle).apply(correct_angle)
                output['head_horizontal_offset'] = (
                        output.target_horizontal_angle - output.head_horizontal_angle).apply(correct_angle)
                output['head_vertical_offset'] = (
                        output.target_vertical_angle - output.head_vertical_angle).apply(correct_angle)
                output['eyeRay_horizontal_offset'] = (
                        output.target_horizontal_angle - output.eyeRay_horizontal_angle).apply(correct_angle)
                output['eyeRay_vertical_offset'] = (
                        output.target_vertical_angle - output.eyeRay_vertical_angle).apply(correct_angle)
                success_record = f.name[-14:-5]
                return output, success_record


def asSpherical(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return [0, 0]
    theta = math.degrees(math.acos(y / r))
    # phi  = math.degrees(math.atan(y/x))
    phi = math.degrees(math.atan2(x, z))
    return [theta, phi]

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