import itertools
import json
import os

import time
from pathlib import Path

import demjson
import numpy as np
import pandas as pd
from pandas import DataFrame

from QuaternionHandling import quaternion_to_euler


# ROOT = Path.cwd()
# DATA_ROOT = ROOT / 'data'
def logging_time(original_fn):
    """[summary]
    Args:
        original_fn ([function]): [function that you want to know how much time does it takes]
    """

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(
            "Working time[{}] : {} sec".format(
                original_fn.__name__, end_time - start_time
            )
        )
        return result

    return wrapper_fn


def read_eye_file(
    target: int, environment: str, block: int, subject: int
) -> pd.DataFrame:
    ROOT = Path(__file__).resolve().parent
    DATA_ROOT = ROOT / "data" / (str(subject))
    filename = make_trial_info(target, environment, block)
    try:
        refined_files = DATA_ROOT.rglob(("EYE*" + str(subject) + "*.pkl"))
        for file in refined_files:
            if filename in file.name:
                output = pd.read_pickle(DATA_ROOT / file.name)
                return output
        files = DATA_ROOT.rglob("EYE*" + str(subject) + "*.csv")
        for file in files:
            if filename in file.name:
                output = pd.read_csv(DATA_ROOT / file.name, header=1)
                refined = manipulate_eye(output)
                refined = refined.astype({"theta": float, "phi": float})
                refined[["norm_x", "norm_y"]] = pd.DataFrame(
                    refined.norm_pos.tolist(), index=refined.index
                )
                refined.to_pickle(DATA_ROOT / (file.name.split(".")[0] + ".pkl"))

                return refined
    except:
        print("error in reading eye file", subject, target, environment, block)
        pass


@logging_time
def manipulate_eye(_eye_dataframe: pd.DataFrame):
    """[summary]

    Args:
        _eye_dataframe (pd.DataFrame): pandas dataframe of raw eye-data 

    Raises:
        ValueError: If there is an error while handling eye data, raises error

    Returns:
        [pd.DataFrame]: 1) reset timestamp to 0, 2) make json list to dataframe
    """
    try:
        eye_list = []
        for row in _eye_dataframe.itertuples(index=False):
            python_timestamp = row[0]
            pupil_data = row[1]
            json_dict = demjson.decode(pupil_data)
            json_dict["python_timestamp"] = python_timestamp
            json_dict["timestamp"] = float(json_dict["timestamp"])
            json_dict["confidence"] = float(json_dict["confidence"])
            json_dict["theta"] = float(json_dict["theta"])
            json_dict["phi"] = float(json_dict["phi"])
            eye_list.append(json_dict)
        output = pd.DataFrame(eye_list)
        # print(output.timestamp[:50])
        output.timestamp = output.timestamp - output.timestamp[0]
        return output
    except:
        print("failed in eye manipulation")
        # raise ValueError("fail in EYE manipulation")


def filter_out_eye(_eye_dataframe: pd.DataFrame, threshold=0.6):
    """[summary]

    Args:
        _eye_dataframe (pd.DataFrame): manipulated eye-dataframe
        threshold (float, optional): confidence threshold, remove lines have below threshold. Defaults to 0.6.

    Returns:
        str: "ok" if it has no critical flaw, "short" for less than ~75%, "low" for too many low confidence lines
    """
    if _eye_dataframe.shape[0] < 600:
        print(f"too short data length {_eye_dataframe.shape[0]}")
        return "short"
    else:
        if _eye_dataframe[_eye_dataframe["confidence"] > threshold].shape[0] < 600:
            print(
                f"too low confidence {_eye_dataframe[_eye_dataframe['confidence'] > threshold].shape[0]}"
            )
            return "low"
    return "ok"


def read_imu_file(
    target: int, environment: str, block: int, subject: int
) -> pd.DataFrame:
    ROOT = Path(__file__).resolve().parent
    DATA_ROOT = ROOT / "data" / (str(subject))
    filename = make_trial_info(target, environment, block)
    try:
        files = DATA_ROOT.rglob("IMU*" + str(subject) + "*.csv")
        for file in files:
            if filename in file.name:
                output = pd.read_csv(DATA_ROOT / file.name, header=1)
                refined = manipulate_imu(output)
                return refined
    except:
        pass


def manipulate_imu(_imu_dataframe: pd.DataFrame):
    """convert raw imu data (quaternion) to euler angles, and set timestamp unit into second.

    Args:
        _imu_dataframe (pd.DataFrame): raw data from csv file

    Raises:
        ValueError: notify there is an error

    Returns:
        [pd.Dataframe]: refined pandas dataframe
    """
    angle_list = []
    try:
        # Change quaternion to euler angles
        for row in _imu_dataframe.itertuples(index=False):
            euler_angle = quaternion_to_euler(row[1], row[2], row[3], row[4])
            x = -euler_angle[0] if (-euler_angle[0] < 180) else -euler_angle[0] - 360
            y = -euler_angle[1] if (-euler_angle[1] < 180) else -euler_angle[1] - 360
            z = (
                -euler_angle[2] - 180
                if (-euler_angle[2] - 180 > -180)
                else -euler_angle[2] + 180
            )

            euler_angle = (x, y, z)
            output = (int(row[0][1:]),) + euler_angle

            angle_list.append(output)
        output = pd.DataFrame(
            data=angle_list,
            columns=["IMUtimestamp", "rotationX", "rotationY", "rotationZ"],
        )
        # Set timestamp unit (ms -> s)
        output.IMUtimestamp = (output.IMUtimestamp - output.IMUtimestamp[0]) / 1000
        return output
    except:
        raise ValueError("fail in IMU manipulation")


def read_hololens_json(
    target: int, environment: str, block: int, subject: int
) -> pd.DataFrame:
    """
    Read a json file from hololens, convert into pandas dataframe after check there is no error.
    :rtype: pandas.Dataframe
    :param target: number of target (0~7)
    :param environment: environmental setting('U': UI or 'W': World)
    :param block: number of repetition (0~4)
    :param subject: number of participant
    :return: pandas dataframe, None if there is an error
    """
    ROOT = Path(__file__).resolve().parent
    DATA_ROOT = ROOT / "data" / (str(subject) + "_holo")
    filename = make_trial_info(target, environment, block)
    try:
        files = DATA_ROOT.rglob("*.json")
        for file in files:
            if filename in file.name:

                with open(file) as f:
                    output: DataFrame = pd.DataFrame(json.load(f)["data"])
                    return output
    except:
        print("Error: while finding hololens file")
    print("Cannot find the file... return None", str(subject) + ":" + filename)
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
    output = np.array([_dict["x"], _dict["y"], _dict["z"]])
    return output


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle


def refining_hololens_dataframe(_data: pd.DataFrame) -> pd.DataFrame:
    # Initialization of timestamp (set start-point to 0)
    _data.timestamp = _data.timestamp - _data.timestamp[0]
    # Deserialize Vector3 components
    for col, item in itertools.product(
        ["head_position", "head_rotation", "head_forward", "target_position"],
        ["x", "y", "z"],
    ):
        _data[col + "_" + item] = _data[col].apply(pd.Series)[item]
    # Change angle range to -180 ~ 180
    for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
        _data[col] = _data[col].apply(change_angle)

    return _data


def bring_hololens_data(
    target: int, environment: str, block: int, subject: int
) -> pd.DataFrame:
    try:
        df = read_hololens_json(target, environment, block, subject)
        return refining_hololens_dataframe(df)
    except:
        pass


if __name__ == "__main__":
    subjects = range(411, 412)
    envs = ["W"]
    targets = range(8)
    blocks = range(5)
    for subject, env, target, block in itertools.product(
        subjects, envs, targets, blocks
    ):
        ROOT = Path(__file__).resolve().parent
        DATA_ROOT = ROOT / "data" / (str(subject))
        files = DATA_ROOT.rglob("*.csv")
        for file in files:
            print(file, " ---> ", DATA_ROOT / (file.name.replace("411", "310")))
            os.rename(file, DATA_ROOT / (file.name.replace("411", "310")))
