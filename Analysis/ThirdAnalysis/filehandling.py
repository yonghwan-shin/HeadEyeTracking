import json
import os
import os.path
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

from pandas import DataFrame

ROOT = Path.cwd()
DATA_ROOT = ROOT / 'data'


def read_hololens_json(target: int, environment: str, block: int, subject: int) -> pd.DataFrame:
    """
    Read a json file from hololens, convert into pandas dataframe after check there is no error.
    :rtype: pandas.Dataframe
    :param target: number of target (0~7)
    :param environment: environmental setting('U': UI or 'W': World)
    :param block: number of repetition (0~4)
    :param subject: number of participant
    :return: pandas dataframe, None if there is an error
    """
    filename = make_trial_info(target, environment, block)
    try:
        files = DATA_ROOT.rglob('#NEXT*S' + str(subject) + '*.json')
        for file in files:
            if filename in file.name:
                with open(file) as f:
                    output: DataFrame = pd.DataFrame(json.load(f)['data'])
                    return output
    except IOError as e:
        print("Error: while finding hololens file", e)
    print("Cannot find the file... return None", filename)
    return pd.DataFrame()


def make_trial_info(target, environment, block):
    """
    Build filename from trial's detail (which target, environment, block)
    :param target: number of target (0~7)
    :param environment: environmental setting( 'U' for UI or 'W' for World)
    :param block: number of repetition (0~4)
    :return: complete string of trial details that should be contained in full filename
    """
    output = "T" + str(target) + "_E" + str(environment) + "_B" + str(block)
    return output


def dict_to_vector(_dict: dict):
    output = np.array([_dict['x'], _dict['y'], _dict['z']])
    return output


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle


def refining_hololens_dataframe(_data: pd.DataFrame) -> pd.DataFrame:
    # Initialization of timestamp (set start-point to 0)
    _data.timestamp = _data.timestamp - _data.timestamp[0]
    # Deserialize Vector3 components
    for col, item in itertools.product(['head_position', 'head_rotation', 'head_forward', 'target_position'],
                                       ['x', 'y', 'z']):
        _data[col + '_' + item] = _data[col].apply(pd.Series)[item]
    # Change angle range to -180 ~ 180
    for col in ['head_rotation_x', 'head_rotation_y', 'head_rotation_z']:
        _data[col] = _data[col].apply(change_angle)

    return _data
