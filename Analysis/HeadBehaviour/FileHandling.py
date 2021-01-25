# %%
from pathlib import Path
import pandas as pd
import json
import itertools
import math
import numpy as np
import time
from scipy import interpolate, signal


# %%


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle


def angle_velocity(_head_forward, _head_forward2, _time):
    import vg

    if type(_head_forward2) is None:
        return None
    vector1 = np.array([_head_forward[0], _head_forward[1], _head_forward[2]])
    vector2 = np.array([_head_forward2[0], _head_forward2[1], _head_forward2[2]])
    return vg.angle(vector1, vector2) / _time


def asSpherical(xyz: list):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.acos(z / r) * 180 / math.pi  # to degrees
    phi = math.atan2(y, x) * 180 / math.pi
    return [r, theta, phi]


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


def crosscorr(datax, datay, lag=0, wrap=False):
    """

    Args:
        datax: base array to compare
        datay: second array to compare
        lag: how many rows to shift
        wrap: wrap up the outside array if True. Defaults to False

    Returns:
        List: list of correlation results
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def synchronise_timestamp(imu, holo, show_plot=False):
    """synchronise Hololens <--> imu timestamp with correlating horizontal values

    Args:
        imu (pd.DataFrame): IMU dataframe
        holo (pd.DataFrame): Hololens dataframe
        show_plot (bool): shows plot of correlation values if True. Defaults to False.

    Returns:
        int,float,float: shift, correlation coef,shifted time(add to imu timestamp)
    """
    time_max = min(holo.timestamp.values[-1], imu.timestamp.values[-1])
    # holo = holo[holo.timestamp <= time_max]
    imu = imu[imu.timestamp <= time_max]
    holo_intp = interpolate.interp1d(holo.timestamp, holo.head_rotation_y)
    holo_interpolated = pd.Series(holo_intp(imu.timestamp))
    approx_range = np.arange(-20, 0)
    rsx = [
        crosscorr(pd.Series(signal.detrend(holo_interpolated)),
                  pd.Series(signal.detrend(imu.rotationZ)), lag)
        for lag in approx_range
    ]
    shift = approx_range[np.argmax(rsx)]
    coef = rsx[int(np.argmax(rsx))]
    shift_time = imu.timestamp.iloc[-1] - imu.timestamp.iloc[shift]
    if show_plot:
        pass
        # _, ax = plt.subplots(figsize=(14, 3))
        # ax.plot(approx_range, rsx)
        # ax.axvline(shift, color='r', linestyle='--')
        # plt.show()

    return shift, coef, shift_time


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


def read_imu_data(target, environment, posture, block, subject, study_num) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent.parent / 'Datasets'
    if study_num == 2:  # FIXME : didn't applied on 2nd dta
        subject += 200
        data_root = root / '2ndData' / 'hololens_data' / ('compressed_sub' + str(subject))
        trial_detail = "T" + str(target) + "_E" + environment + "_P" + posture + "_B" + str(block)
    elif study_num == 3:
        subject += 300
        data_root = root / '3rdData' / (str(subject))
        trial_detail = "T" + str(target) + "_E" + environment + "_B" + str(block)
        try:
            pickled_files = data_root.rglob('IMU*' + trial_detail + '*.pkl')
            for file in pickled_files:
                if trial_detail in file.name:
                    return pd.read_pickle(file)
            whole_files = data_root.rglob('IMU*' + trial_detail + '*.csv')
            for file in whole_files:
                if trial_detail in file.name:
                    with open(file) as f:
                        # third_output = pd.DataFrame(json.load(f)['data'])
                        third_output = pd.read_csv(f, header=1)
                        output = manipulate_imu(third_output)
                        output = output.rename(columns={"IMUtimestamp": "timestamp"})
        except Exception as e:
            raise Exception('----Finding IMU error-----\n', e.args)
    else:
        print('study_num should be 2 or 3, input was ', study_num)
    return output


def read_eye_data(target, environment, posture, block, subject, study_num):
    root = Path(__file__).resolve().parent.parent.parent / 'Datasets'
    if study_num == 2:  # FIXME : didn't applied on 2nd dta
        subject += 200
        data_root = root / '2ndData' / 'hololens_data' / ('compressed_sub' + str(subject))
        trial_detail = "T" + str(target) + "_E" + environment + "_P" + posture + "_B" + str(block)
    elif study_num == 3:
        subject += 300
        data_root = root / '3rdData' / (str(subject))
        trial_detail = "T" + str(target) + "_E" + environment + "_B" + str(block)
        try:
            pickled_files = data_root.rglob('EYE*' + trial_detail + '*.pkl')
            for file in pickled_files:
                if trial_detail in file.name:
                    third_output = pd.read_pickle(file)
                    third_output.timestamp = third_output.timestamp - third_output.timestamp[0]
                    third_output.python_timestamp = third_output.python_timestamp - third_output.python_timestamp[0]
                    third_output = third_output.astype({
                        'theta': float,
                        'phi': float,
                        'norm_x': float,
                        'norm_y': float
                    })
                    output=third_output
                    return output
            whole_files = data_root.rglob('EYE*' + trial_detail + '*.csv')
            for file in whole_files:
                if trial_detail in file.name:
                    with open(file) as f:
                        third_output = pd.DataFrame(f,header=1)
                        third_output.timestamp = third_output.timestamp - third_output.python_timestamp[0]
                        third_output.python_timestamp = third_output.python_timestamp - third_output.python_timestamp[0]
                        third_output = third_output.astype({
                            'theta': float,
                            'phi': float,
                            'norm_x': float,
                            'norm_y': float
                        })
                        third_output.norm_x = third_output.norm_x.astype(float)
                        third_output.norm_y = third_output.norm_y.astype(float)
                        output = third_output
        except Exception as e:
            raise Exception('----Finding EYE error-----\n', e.args)
    else:
        print('study_num should be 2 or 3, input was ', study_num)
    return output


def read_hololens_data(target, environment, posture, block, subject, study_num):
    root = Path(__file__).resolve().parent.parent.parent / 'Datasets'
    if study_num == 2:
        subject += 200
        data_root = root / '2ndData' / 'hololens_data' / ('compressed_sub' + str(subject))
        trial_detail = "T" + str(target) + "_E" + environment + "_P" + posture + "_B" + str(block)
        try:
            pickled_files = data_root.rglob('*' + trial_detail + '*.pkl')
            for file in pickled_files:
                if trial_detail in file.name:
                    return pd.read_pickle(file)
            whole_files = data_root.rglob('*' + trial_detail + '*.csv')
            for file in whole_files:
                if trial_detail in file.name:
                    second_output = pd.read_csv(file, index_col=False, header=1)
                    second_output = second_output.rename(columns={
                        'Timestamp': 'timestamp',
                        'HeadPositionX': 'head_position_x',
                        'HeadPositionY': 'head_position_y',
                        "HeadPositionZ": 'head_position_z',
                        'HeadRotationX': 'head_rotation_x',
                        'HeadRotationY': 'head_rotation_y',
                        'HeadRotationZ': 'head_rotation_z',
                        'HeadForwardX': 'head_forward_x',
                        'HeadForwardY': 'head_forward_y',
                        'HeadForwardZ': 'head_forward_z',
                        'TargetPositionX': 'target_position_x',
                        'TargetPositionY': 'target_position_y',
                        'TargetPositionZ': 'target_position_z',
                        'TargetEntered': 'target_entered',
                        'TargetAngularDistance': 'angular_distance'
                    })
                    second_output.timestamp = second_output.timestamp - second_output.timestamp[0]
                    # Change angle range to -180 ~ 180
                    for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
                        second_output[col] = second_output[col].apply(change_angle)
                    output = second_output
        except Exception as e:
            raise Exception('-----Finding Hololens error-----\n', e.args)

    elif study_num == 3:
        subject += 300
        data_root = root / '3rdData' / (str(subject) + '_holo')
        trial_detail = "T" + str(target) + "_E" + environment + "_B" + str(block)
        try:
            pickled_files = data_root.rglob('*' + trial_detail + '*.pkl')
            for file in pickled_files:
                if trial_detail in file.name:
                    return pd.read_pickle(file)
            whole_files = data_root.rglob('*' + trial_detail + '*.json')
            for file in whole_files:
                if trial_detail in file.name:
                    with open(file) as f:
                        third_output = pd.DataFrame(json.load(f)['data'])
                        third_output.timestamp = third_output.timestamp - third_output.timestamp[0]
                        for col, item in itertools.product(
                                ["head_position", "head_rotation", "head_forward", "target_position"],
                                ["x", "y", "z"],
                        ):
                            third_output[col + "_" + item] = third_output[col].apply(pd.Series)[item]
                        # Change angle range to -180 ~ 180
                        for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
                            third_output[col] = third_output[col].apply(change_angle)
                        third_output = third_output.drop(
                            ['head_position', 'head_rotation', 'head_forward', 'target_position'], axis=1)
                        output = third_output
        except Exception as e:
            raise Exception('----Finding Hololens error-----\n', e.args)
    else:
        print('study_num should be 2 or 3, input was ', study_num)

    output['head_forward_x_next'] = output.head_forward_x.shift(1)
    output['head_forward_y_next'] = output.head_forward_y.shift(1)
    output['head_forward_z_next'] = output.head_forward_z.shift(1)
    output['time_interval'] = output.timestamp.diff()
    output['angle_speed'] = output.apply(
        lambda row: angle_velocity([row.head_forward_x, row.head_forward_y, row.head_forward_z],
                                   [row.head_forward_x_next, row.head_forward_y_next, row.head_forward_z_next],
                                   row.time_interval
                                   ), axis=1
    )
    thetas = []
    phis = []
    distances = []
    maxTargetsizes = []
    for index, row in output.iterrows():
        x = row["target_position_x"] - row["head_position_x"]
        y = row["target_position_y"] - row["head_position_y"]
        z = row["target_position_z"] - row["head_position_z"]
        # x, y = apply_zaxis(x, y, row['head_rotation_z'])
        [r, theta, phi] = asSpherical([x, z, y])
        distances.append(r)
        thetas.append(-90 + theta)
        phis.append(90 - phi)
        maxTargetsizes.append(r * math.sin(math.radians(row['angular_distance'])))
    output['Distance'] = distances
    output['Theta'] = thetas
    output["Phi"] = phis
    output['TargetVertical'] = output.head_rotation_x - output.Theta
    output['TargetHorizontal'] = output.head_rotation_y - output.Phi
    output['MaximumTargetSize'] = maxTargetsizes
    output.to_pickle(data_root / (str(trial_detail) + '.pkl'))
    return output


if __name__ == '__main__':

    subjects = range(1, 17)
    envs = ['U', 'W']
    targets = range(8)
    blocks = range(1, 5)
    final_result = []
    start = time.time()
    for subject, env, target, block in itertools.product(
            subjects, envs, targets, blocks
    ):
        try:
            print(subject, env, target, block)
            output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                        study_num=3)
        except Exception as e:
            print(e)
