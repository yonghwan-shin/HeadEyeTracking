import pandas as pd
from pathlib import Path
import json


def read_hololens_data(subject, posture, cursor_type, repetition):
    root = Path(__file__).resolve().parent / 'data'
    trial_detail = f'subject{str(subject)}_posture{str(posture)}_cursor{str(cursor_type)}_repetition{str(repetition)}'
    files = root.rglob('*' + trial_detail + '*.json')
    try:
        for file in files:
            if trial_detail in file.name:
                with open(file) as f:  # found exact file
                    output = pd.DataFrame(json.load(f))
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
    first_end_num = data['end_num'].values[0]
    for i in range(len(data) - 1):
        if data['end_num'].values[i] == first_end_num:
            pass
            # data = data.drop(i)
            # print(first_end_num, 'drop', i, len(data))
        else:
            # print(i, data['timestamp'].values[i],data['end_num'].values[i],first_end_num)
            data=data.drop([x for x in range(i)])
            break
    for target_num in range(9):
        output.append(data[data['end_num'] == target_num])
    return output
