import itertools
import json
import os
import time
from pathlib import Path
import math
import demjson
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import interpolate
from QuaternionHandling import *


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
        print("Working time[{}] : {} sec".format(original_fn.__name__,
                                                 end_time - start_time))
        return result

    return wrapper_fn


def asSpherical(xyz: list):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.acos(z / r) * 180 / math.pi  # to degrees
    phi = math.atan2(y, x) * 180 / math.pi
    return [r, theta, phi]


def read_eye_file(target: int, environment: str, block: int,
                  subject: int) -> pd.DataFrame:
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
                # refined = refined.astype({"theta": float, "phi": float})
                refined[["norm_x",
                         "norm_y"]] = pd.DataFrame(refined.norm_pos.tolist(),
                                                   index=refined.index)
                refined = refined.astype({
                    "theta": float,
                    "phi": float,
                    "norm_x": float,
                    "norm_y": float
                })
                refined.to_pickle(DATA_ROOT /
                                  (file.name.split(".")[0] + ".pkl"))

                return refined
    except Exception as e:
        raise Exception(e.args, "error in reading eye file", subject, target,
                        environment, block)


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
    except Exception as e:
        raise Exception(e.args, "failed in eye manipulation")
        # raise ValueError("fail in EYE manipulation")


def check_eye_dataframe(_eye_dataframe: pd.DataFrame, threshold=0.6):
    """[summary]

    Args:
        _eye_dataframe (pd.DataFrame): manipulated eye-dataframe
        threshold (float, optional): confidence threshold, remove lines have below threshold. Defaults to 0.6.

    Returns:
        str: "ok" if it has no critical flaw, "short" for less than ~75%, "low" for too many low confidence lines
    """
    if _eye_dataframe.shape[0] < 600:
        raise Exception(f"too short data length {_eye_dataframe.shape[0]}",
                        "short")

    else:
        if _eye_dataframe[
            _eye_dataframe["confidence"] > threshold].shape[0] < 600:
            raise Exception(
                f"too low confidence {_eye_dataframe[_eye_dataframe['confidence'] > threshold].shape[0]}",
                "low",
            )


def check_hololens_dataframe(_holo_dataframe: pd.DataFrame,
                             block,
                             threshold=4.0):
    """Check the hololens dataframe is properly done
    Args:
        _holo_dataframe (pd.DataFrame):basic dataframe to check
        block (int): need a number of block (0 means practice trial)
        threshold (float): walklegnth threshold Defaults to 4.0.

    Raises:
        Exception: "short" if it didn't exceed threshold
    """
    walklength = _holo_dataframe.head_position_z.iloc[-1] - \
                 _holo_dataframe.head_position_z.iloc[0]
    if walklength < threshold:
        logstring = ""
        if block == 0:
            logstring = "practice"
        else:
            logstring = "short"
        raise Exception(logstring, walklength)


def read_imu_file(target: int, environment: str, block: int,
                  subject: int) -> pd.DataFrame:
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


# %% Bring Data

def bring_data(target, env, block, subject):
    try:
        holo = bring_hololens_data(target, env, block, subject)

        imu = read_imu_file(target, env, block, subject)
        eye = read_eye_file(target, env, block, subject)
        eye = eye.astype({
            "theta": float,
            "phi": float,
            "norm_x": float,
            "norm_y": float
        })
        return holo, imu, eye
    except:
        print("error in bringing data")


def interpolated_dataframes(holo, imu, eye):
    timestamp = np.arange(0, 6.5, 1 / 120)
    interpolated_holo = pd.DataFrame()
    interpolated_imu = pd.DataFrame()
    interpolated_eye = pd.DataFrame()
    holo_columns = ['angular_distance',
                    'head_position_x', 'head_position_y', 'head_position_z',
                    'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
                    'head_forward_x', 'head_forward_y', 'head_forward_z', 'Theta', 'Phi',
                    'target_position_x', 'target_position_y', 'target_position_z',
                    'norm_target_vector_x', 'norm_target_vector_y', 'norm_target_vector_z',
                    'norm_head_vector_x', 'norm_head_vector_y', 'norm_head_vector_z', 'TargetVertical',
                    'TargetHorizontal']
    for column in holo_columns:
        interpolate_function = interpolate.interp1d(holo.timestamp, holo[column], fill_value='extrapolate')
        interpolated_column = interpolate_function(timestamp)
        interpolated_holo[column] = interpolated_column
    interpolated_holo['timestamp'] = timestamp
    imu_columns = ['rotationX', 'rotationY', 'rotationZ']
    for column in imu_columns:
        interpolate_function = interpolate.interp1d(imu.IMUtimestamp, imu[column], fill_value='extrapolate')
        interpolated_column = interpolate_function(timestamp)
        interpolated_imu[column] = interpolated_column
    interpolated_imu['timestamp'] = timestamp
    eye_columns = ['confidence', 'theta', 'phi', 'norm_x', 'norm_y']
    for column in eye_columns:
        interpolate_function = interpolate.interp1d(eye.timestamp, eye[column], fill_value='extrapolate')
        interpolated_column = interpolate_function(timestamp)
        interpolated_eye[column] = interpolated_column
    interpolated_eye['timestamp'] = timestamp
    return interpolated_holo, interpolated_imu, interpolated_eye


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
            x = -euler_angle[0] if (
                    -euler_angle[0] < 180) else -euler_angle[0] - 360
            y = -euler_angle[1] if (
                    -euler_angle[1] < 180) else -euler_angle[1] - 360
            z = (-euler_angle[2] - 180 if
                 (-euler_angle[2] - 180 > -180) else -euler_angle[2] + 180)

            euler_angle = (x, y, z)
            output = (int(row[0][1:]),) + euler_angle

            angle_list.append(output)
        output = pd.DataFrame(
            data=angle_list,
            columns=["IMUtimestamp", "rotationX", "rotationY", "rotationZ"],
        )
        # Set timestamp unit (ms -> s)
        output.IMUtimestamp = (output.IMUtimestamp -
                               output.IMUtimestamp[0]) / 1000
        return output
    except Exception as e:
        raise Exception(e.args, "fail in IMU manipulation")


def read_hololens_json(target: int, environment: str, block: int,
                       subject: int) -> pd.DataFrame:
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
    except Exception as e:
        raise Exception(e.args, "Error: while finding hololens file")
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


def apply_zaxis(x, y, theta_z):
    # takes list xyz (single coord)
    r = math.sqrt(x * x + y * y)
    beta = math.radians(-theta_z)
    changed_x = math.cos(beta) * x - math.sin(beta) * y
    changed_y = math.sin(beta) * x + math.cos(beta) * y
    # changed_x = r * ((x / r) * (math.cos(math.radians(theta_z))) + (y / r) * (math.sin(math.radians(theta_z))))
    # changed_y = r*((y / r) * (math.cos(math.radians(theta_z))) - (x / r) * (math.sin(math.radians(theta_z))))
    # theta = math.acos(z / r) * 180 / math.pi  # to degrees
    # phi = math.atan2(y, x) * 180 / math.pi
    return changed_x, changed_y


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

    norm_target_vectors = []
    norm_head_vectors = []
    thetas = []
    phis = []
    for index, row in _data.iterrows():
        norm_head_vectors.append(
            np.array([row['head_forward_x'], row['head_forward_y'], row['head_forward_z']]) / row['head_forward_z'])
        x = row["target_position_x"] - row["head_position_x"]
        y = row["target_position_y"] - row["head_position_y"]
        z = row["target_position_z"] - row["head_position_z"]
        # x, y = apply_zaxis(x, y, row['head_rotation_z'])
        [r, theta, phi] = asSpherical([x, z, y])
        thetas.append(90 - theta)
        phis.append(90 - phi)
        norm_target_vector = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
        norm_target_vectors.append(norm_target_vector / norm_target_vector[2])
    _data['norm_head_vector'] = norm_head_vectors
    _data['norm_head_vector_x'] = _data['norm_head_vector'].apply(pd.Series)[0]
    _data['norm_head_vector_y'] = _data['norm_head_vector'].apply(pd.Series)[1]
    _data['norm_head_vector_z'] = _data['norm_head_vector'].apply(pd.Series)[2]
    _data['Theta'] = thetas
    _data["Phi"] = phis
    _data['TargetVertical'] = _data.head_rotation_x + _data.Theta
    _data['TargetHorizontal'] = _data.head_rotation_y - _data.Phi
    _data['norm_target_vector'] = norm_target_vectors
    _data['norm_target_vector_x'] = _data['norm_target_vector'].apply(pd.Series)[0]
    _data['norm_target_vector_y'] = _data['norm_target_vector'].apply(pd.Series)[1]
    _data['norm_target_vector_z'] = _data['norm_target_vector'].apply(pd.Series)[2]
    # for col, item in itertools.product(
    #         ["norm_target_vector"],
    #         ["x", "y", "z"],
    # ):
    #     _data[col + "_" + item] = _data[col].apply(pd.Series)[item]
    return _data


def bring_hololens_data(target: int, environment: str, block: int,
                        subject: int) -> pd.DataFrame:
    try:
        df = read_hololens_json(target, environment, block, subject)
        return refining_hololens_dataframe(df)
    except Exception as e:
        raise Exception(e.args)


if __name__ == "__main__":
    pass
