import json
import os
import os.path
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
                    output.timestamp = output.timestamp - output.timestamp[0]
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
