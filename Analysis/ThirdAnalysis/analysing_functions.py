import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
from plotly.subplots import make_subplots
from scipy import interpolate, signal, stats

from filehandling import *


# %% check the distribution fo x-pos and angle distance
def check_distribution():
    # from plotly.subplots import make_subplots
    # from analysing_functions import *
    # # IF you are using Pycharm
    # import plotly.io as pio
    # from scipy import fftpack, signal

    pio.renderers.default = "browser"
    subjects = range(301, 317)
    envs = ["W", "U"]
    targets = range(8)
    blocks = range(1, 5)

    W_startpositionX = []
    W_startAngles = []
    W_walklengths = []
    U_startpositionX = []
    U_startAngles = []
    U_walklengths = []
    for subject in subjects:
        for env, target, block in itertools.product(["W"], targets, blocks):
            try:
                holo = bring_hololens_data(target, env, block, subject)

                W_startpositionX.append(holo.iloc[5]["head_position_x"])
                W_startAngles.append(holo.iloc[5]["angular_distance"])
                walk = (
                        holo["head_position_z"].iloc[-1] - holo["head_position_z"].iloc[0]
                )
                W_walklengths.append(walk)
                print(env, target, block, subject)
                pass
            except Exception as e:
                print(e.args)
    for subject in subjects:
        for env, target, block in itertools.product(["U"], targets, blocks):
            try:
                holo = bring_hololens_data(target, env, block, subject)

                U_startpositionX.append(holo.iloc[5]["head_position_x"])
                U_startAngles.append(holo.iloc[5]["angular_distance"])
                walk = (
                        holo["head_position_z"].iloc[-1] - holo["head_position_z"].iloc[0]
                )
                U_walklengths.append(walk)
                print(env, target, block, subject)
                pass
            except Exception as e:
                print(e.args)
    from scipy.stats import ks_2samp, wasserstein_distance

    print(ks_2samp(W_startpositionX, W_startAngles))
    print(ks_2samp(W_startpositionX, U_startAngles))
    print(ks_2samp(W_startpositionX, U_startpositionX))
    print(ks_2samp(W_startpositionX, W_startpositionX))
    print(ks_2samp(U_startpositionX, W_startAngles))
    print(ks_2samp(U_startpositionX, U_startAngles))
    print(ks_2samp(U_startpositionX, U_startpositionX))
    print(ks_2samp(U_startpositionX, W_startpositionX))

    # #%%
    # import plotly.figure_factory as ff
    #
    # fig = ff.create_distplot([W_startpositionX,U_startpositionX],['W','U'],bin_size=0.02,show_rug=False)
    # fig.show()
    # W_startAngles = pd.Series(W_startAngles)
    # W_startAngles = W_startAngles[W_startAngles<150]
    # fig = ff.create_distplot([W_startAngles,U_startAngles],['W','U'],bin_size=0.2,show_rug=False)
    # fig.show()


def angle_velocity(_head_forward, _head_forward2, _time):
    import vg

    if type(_head_forward2) is not dict:
        return None
    vector1 = np.array([_head_forward["x"], _head_forward["y"], _head_forward["z"]])
    vector2 = np.array([_head_forward2["x"], _head_forward2["y"], _head_forward2["z"]])
    return vg.angle(vector1, vector2) / _time


# subjects = range(307, 308)

def summary_dwell():
    global subjects, envs, targets, blocks, thresholds, subject, env, target, block, holo, summary2
    subjects = range(301, 317)
    envs = ["U", "W"]
    targets = range(8)
    blocks = range(1, 5)
    # threshold = 0.2
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    dwell_output = []
    for subject, env, target, block, threshold in itertools.product(
            subjects, envs, targets, blocks, thresholds
    ):
        print(target, env, block, subject, threshold)
        holo = bring_hololens_data(target, env, block, subject)
        holo["head_forward_next"] = holo.head_forward.shift(1)
        holo["time_interval"] = holo.timestamp.diff()
        holo["angle_speed"] = holo.apply(
            lambda x: angle_velocity(x.head_forward, x.head_forward_next, x.time_interval),
            axis=1,
        )

        success = pd.DataFrame(
            holo.target_entered,
        )
        success["next_frame"] = success.target_entered.shift(1)
        success["start"] = (success.target_entered == True) & (success.next_frame == False)
        success["end"] = (success.target_entered == False) & (success.next_frame == True)

        starts = np.array(success.start[success.start == True].index)
        ends = np.array(success.end[success.end == True].index)
        if starts[-1] > ends[-1]:
            starts = np.delete(starts, -1)
        if starts[0] > ends[0]:
            ends = np.delete(ends, -1)
        durations = ends - starts
        freq = 60

        dwell = np.argwhere(durations > threshold * freq)
        dwell = [x[0] for x in dwell]

        _no_over = True if len(starts) == 0 else False
        if _no_over == True:
            print("no over")
            dwell_output.append(
                dict(
                    subject=subject,
                    environment=env,
                    target=target,
                    block=block,
                    threshold=threshold,
                    no_over=_no_over,
                )
            )
            continue
            # check there is an over-target
        if len(dwell) <= 0:
            print("no dwell")
            dwell_output.append(
                dict(
                    subject=subject,
                    environment=env,
                    target=target,
                    block=block,
                    threshold=threshold,
                    dwell_success=False,
                )
            )
            continue
        # BASIC STATS
        _start_time = holo.timestamp[starts[0]]  # When is the first target-over
        _num_over = len(starts)  # How many times of target-over
        duration_times = [
            holo.timestamp[ends[i]] - holo.timestamp[starts[i]] for i in range(len(starts))
        ]
        _total_dur_over = sum(duration_times)
        _mean_dur_over = _total_dur_over / _num_over
        _angle_dist_mean = holo.angular_distance.mean()
        _longest_over = max(duration_times)

        # DWELL Stats
        _dwell_success = True if len(dwell) > 0 else False  # Success rate
        _dwell_time = holo.timestamp[starts[dwell[0]]]  # first dwell timestamp

        # SPEED
        _mean_speeds = []
        _vel_ins = []
        _vel_outs = []
        _last100s = []
        for i in range(len(dwell)):
            _mean_speed = holo.iloc[starts[dwell[i]]: ends[dwell[i]]].angle_speed.mean()
            _vel_in = holo.iloc[starts[dwell[i]]].angle_speed
            _vel_out = holo.iloc[ends[dwell[i]]].angle_speed
            _last100 = holo.iloc[ends[dwell[i]] - 6: ends[dwell[i]]].angle_speed.mean()
            _mean_speeds.append(_mean_speed)
            _vel_ins.append(_vel_in)
            _vel_outs.append(_vel_out)
            _last100s.append(_last100)

        _prior_count = int(dwell[0])

        output = dict(
            subject=subject,
            environment=env,
            target=target,
            block=block,
            threshold=threshold,
            no_over=_no_over,
            start_time=_start_time,
            num_over=_num_over,
            total_dur_over=_total_dur_over,
            mean_dur_over=_mean_dur_over,
            angle_dist_mean=_angle_dist_mean,
            longest_over=_longest_over,
            dwell_success=_dwell_success,
            dwell_time=_dwell_time,
            mean_speeds=_mean_speeds,
            vel_ins=_vel_ins,
            vel_outs=_vel_outs,
            last_100s=_last100s,
            prior_count=_prior_count,
        )
        dwell_output.append(output)
    summary = pd.DataFrame(
        dwell_output,
    )
    # return summary
    summary_sub = dict([])
    for subject, env in itertools.product(subjects, envs):
        summary_sub[(subject, env)] = summary2.loc[
            (summary2["subject"] == subject) & (summary2["environment"] == env)
            ]
    return summary_sub


def list_to_mean(data):
    if type(data) is not list:
        return None
    return sum(data) / len(data)


def make_summary_proceed(summary_sub):
    summary_proceed = []
    for subject, env in itertools.product(subjects, envs):
        data = summary_sub[(subject, env)]
        _mean_start_time = data[data["threshold"] == 0.1].start_time.mean()
        _mean_num_over = data[data["threshold"] == 0.1].num_over.mean()
        _mean_total_dur_over = data[data["threshold"] == 0.1].total_dur_over.mean()
        _mean_dur_over = data[data["threshold"] == 0.1].mean_dur_over.mean()
        _mean_angle_dist = data[data["threshold"] == 0.1].angle_dist_mean.mean()
        _mean_longest_over = data[data["threshold"] == 0.1].longest_over.mean()
        # no over is  None
        dwell_success_count = dict([])
        dwell_time = dict([])
        mean_speeds = dict([])
        prior_counts = dict([])
        last_100s = dict([])
        for threshold in thresholds:
            dwell_success_count["dwell_success_count" + str(threshold)] = data[
                (data["threshold"] == threshold) & (data["dwell_success"] == True)
                ].shape[0]
            dwell_time["dwell_time" + str(threshold)] = data[
                data["threshold"] == threshold
                ].dwell_time.mean()
            mean_speeds["mean_speeds" + str(threshold)] = (
                data[data["threshold"] == threshold]
                    .mean_speeds.apply(list_to_mean)
                    .mean()
            )
            prior_counts["prior_counts" + str(threshold)] = data[
                data["threshold"] == threshold
                ].prior_count.mean()
            last_100s["last_100s" + str(threshold)] = (
                data[data["threshold"] == threshold]
                    .last_100s.apply(list_to_mean)
                    .mean()
            )
        output = dict(
            subject=subject,
            environment=env,
            mean_start_time=_mean_start_time,
            mean_num_over=_mean_num_over,
            mean_total_dur_over=_mean_total_dur_over,
            mean_dur_over=_mean_dur_over,
            mean_angle_dist=_mean_angle_dist,
            mean_longest_over=_mean_longest_over,
            dwell_success_count=dwell_success_count,
            dwell_time=dwell_time,
            mean_speeds=mean_speeds,
            prior_counts=prior_counts,
            last_100s=last_100s,
        )
        summary_proceed.append(output)
    df_summary_proceed = pd.DataFrame(summary_proceed)
    for col in [
        "dwell_success_count",
        "dwell_time",
        "mean_speeds",
        "prior_counts",
        "last_100s",
    ]:
        df_summary_proceed = pd.concat(
            [
                df_summary_proceed.drop([col], axis=1),
                df_summary_proceed[col].apply(pd.Series),
            ],
            axis=1,
        )

    df_summary_proceed.to_excel("summary_proceed.xlsx")


# %% test one trial
def summary_one_trial(target, env, block, subject):
    # 0, "U", 4, 316
    holo, imu, eye = bring_data(target, env, block, subject)
    shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
    imu.IMUtimestamp = imu.IMUtimestamp - shift_time
    eye.timestamp = eye.timestamp - shift_time
    # eye.norm_x = eye.norm_x - eye.norm_x.mean()
    # eye.norm_y = eye.norm_y - eye.norm_y.mean()

    # imu = imu_to_vector(imu)
    new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
    new_imu = imu_to_vector(new_imu)
    new_holo = holo_to_vector(new_holo)
    # new_holo = new_holo[new_holo.timestamp >= 1.5]
    # new_imu = new_imu[new_imu.timestamp >= 1.5]
    # new_eye = new_eye[new_eye.timestamp >= 1.5]
    # rot, rmsd = R.align_vectors(new_holo[['head_forward_x', 'head_forward_y', 'head_forward_z']],
    #                             new_imu[['vector_x', 'vector_y', 'vector_z']])
    # applied_imu = rot.apply(new_imu[['vector_x', 'vector_y', 'vector_z']])

    fs = 120
    fc = 4
    w = fc / (fs / 2)
    mincutoff = 3.0
    beta = 0.01
    b, a = signal.butter(3, w, "low", analog=False)
    # filtered_norm_x= pd.Series(signal.filtfilt(b, a, new_eye.norm_x))
    filtered_norm_x = pd.Series(
        one_euro(new_eye.norm_x, beta=beta, mincutoff=mincutoff)
    )
    filtered_norm_x.index = new_eye.index
    new_eye["filtered_norm_x"] = filtered_norm_x
    filtered_norm_y = pd.Series(
        one_euro(new_eye.norm_y, beta=beta, mincutoff=mincutoff)
    )
    filtered_norm_y.index = new_eye.index
    new_eye["filtered_norm_y"] = filtered_norm_y
    filtered_target_horizontal = pd.Series(
        one_euro(new_holo.TargetHorizontal, beta=beta, mincutoff=mincutoff)
    )
    filtered_target_horizontal.index = new_holo.index
    new_holo["filtered_TargetHorizontal"] = filtered_target_horizontal
    filtered_target_vertical = pd.Series(
        one_euro(new_holo.TargetVertical, beta=beta, mincutoff=mincutoff)
    )
    filtered_target_vertical.index = new_holo.index
    new_holo["filtered_TargetVertical"] = filtered_target_vertical

    # get multiple?
    # multiple_horizontal = interpolated_target_horizontal.abs().mean() / eye.norm_x.abs().mean()
    # multiple_vertical = interpolated_target_vertical.abs().mean() / eye.norm_y.abs().mean()
    offset = int(120 * 1.5)
    multiple_horizontal = (
            new_holo.filtered_TargetHorizontal[offset:].abs().mean()
            / new_eye.filtered_norm_x[offset:].abs().mean()
    )
    multiple_vertical = (
            new_holo.filtered_TargetVertical[offset:].abs().mean()
            / new_eye.filtered_norm_y[offset:].abs().mean()
    )

    # HORIZONTAL
    fig_horizontal = make_subplots(rows=4, cols=1)
    fig_horizontal.update_layout(
        title=dict(text=str(multiple_horizontal), font={"size": 30})
    )
    # EYE
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=(new_eye.norm_x * multiple_horizontal),
            name="eye-x",
            opacity=0.5,
        ),
        row=1,
        col=1,
    )
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=(new_eye.filtered_norm_x * multiple_horizontal),
            name="filtered-eye-x",
        ),
        row=1,
        col=1,
    )
    # HOLO
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.filtered_TargetHorizontal,
            name="filtered_target-x",
        ),
        row=2,
        col=1,
    )
    # slope_horizontal, intercept_horizontal = normalize(new_imu.rotationZ, new_holo.head_rotation_y)
    # fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_imu.rotationZ * slope_horizontal + intercept_horizontal, name='imu-x'), row=2, col=1)
    fig_horizontal.add_trace(
        go.Scatter(
            x=holo.timestamp, y=holo.head_rotation_y - holo.Phi, name="target-x"
        ),
        row=2,
        col=1,
    )
    # TARGET
    # fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetHorizontal, name='target-x'), row=3, col=1)
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.filtered_TargetHorizontal,
            name="filtered-target-x",
        ),
        row=3,
        col=1,
    )
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.TargetHorizontal,
            name="target-x",
            opacity=0.3,
        ),
        row=3,
        col=1,
    )
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=-new_eye.filtered_norm_x * multiple_horizontal,
            name="filtered-eye-x",
        ),
        row=3,
        col=1,
    )
    # COMPENSATED
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.filtered_TargetHorizontal
              + new_eye.filtered_norm_x * multiple_horizontal,
            name="compensated",
        ),
        row=4,
        col=1,
    )

    # VERTICAL
    fig_vertical = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=("eye", "head", "target", "compensation"),
    )
    fig_vertical.update_layout(
        title=dict(text=str(multiple_vertical), font=dict(size=30))
    )
    # EYE
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=(new_eye.norm_y * multiple_vertical),
            name="eye-y",
            opacity=0.5,
        ),
        row=1,
        col=1,
    )
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=(new_eye.filtered_norm_y * multiple_vertical),
            name="filtered-eye-y",
        ),
        row=1,
        col=1,
    )
    # HOLO
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=new_holo.head_rotation_x, name="head-y"),
        row=2,
        col=1,
    )
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.filtered_TargetVertical,
            name="filtered-target-y",
        ),
        row=2,
        col=1,
    )
    # fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_imu.rotationX, name='imu-y'), row=2, col=1)
    fig_vertical.add_trace(
        go.Scatter(
            x=holo.timestamp,
            y=holo.Theta + holo.head_rotation_x,
            name="target-y",
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    # TARGET
    # fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetVertical, name='target-y'), row=3, col=1)
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.filtered_TargetVertical,
            name="filtered-target-y",
        ),
        row=3,
        col=1,
    )
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=(-new_eye.filtered_norm_y * multiple_vertical),
            name="eye-y",
        ),
        row=3,
        col=1,
    )
    # fill='tonexty
    # COMPENSATED
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.TargetVertical + new_eye.norm_y * multiple_vertical,
            name="unfiltered-compensate",
            opacity=0.3,
        ),
        row=4,
        col=1,
    )
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp,
            y=new_holo.filtered_TargetVertical
              + new_eye.filtered_norm_y * multiple_vertical,
            name="compensated-y",
        ),
        row=4,
        col=1,
    )
    fig_vertical.update_yaxes(range=[-2, 2], row=1, col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=2, col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=3, col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=4, col=1)
    window = 120
    estimation_horizontal = []
    estimation_vertical = []
    for index in range(new_eye.shape[0] - window - 1):
        temp_holo = new_holo[index: index + window]
        temp_eye = new_eye[index: index + window]
        slope_horizontal, intercept_horizontal = normalize(
            temp_eye.filtered_norm_x, temp_holo.filtered_TargetHorizontal
        )
        slope_vertical, intercept_vertical = normalize(
            temp_eye.filtered_norm_y, temp_holo.filtered_TargetVertical
        )
        estimated_horizontal = (
                new_eye.iloc[index + window + 1]["filtered_norm_x"] * slope_horizontal
                + intercept_horizontal
        )
        estimated_vertical = (
                new_eye.iloc[index + window + 1]["filtered_norm_y"] * slope_vertical
                + intercept_vertical
        )
        estimation_horizontal.append(estimated_horizontal)
        estimation_vertical.append(estimated_vertical)
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp[window + 1:],
            y=estimation_horizontal,
            name="real-time-x",
        ),
        row=3,
        col=1,
    )
    fig_horizontal.add_trace(
        go.Scatter(
            x=new_eye.timestamp[window + 1:],
            y=new_holo.filtered_TargetHorizontal[window + 1:] - estimation_horizontal,
            name="estimation-x",
        ),
        row=4,
        col=1,
    )
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp[window + 1:], y=estimation_vertical, name="real-time-y"
        ),
        row=3,
        col=1,
    )
    fig_vertical.add_trace(
        go.Scatter(
            x=new_eye.timestamp[window + 1:],
            y=new_holo.filtered_TargetVertical[window + 1:] - estimation_vertical,
            name="estimation-y",
        ),
        row=4,
        col=1,
    )
    fig_horizontal.show()
    fig_vertical.show()


def estimate_eye(_eye, _head, window=60):
    # check the array length
    if (len(_eye) != len(_head)):
        return []
    estimation = []
    slopes = []
    intercepts = []
    for i in range(len(_eye) - window - 1):
        e = _eye[i:i + window]
        h = _head[i:i + window]
        slope, intercept = linear_regression(e, h)
        slopes.append(slope)
        intercepts.append(intercept)
        # multiple, _ = normalize(e,h)
        # output = (_eye[i+1] - e.mean()) * multiple + h.mean()
        output = _eye[i + window + 1] * slope + intercept
        # output = _head.mean() - output
        estimation.append(output)
    return estimation, slopes, intercepts


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


def normalize(_from, _to):
    a = _from - sum(_from)/len(_from)
    b = _to - sum(_to)/len(_to)
    multiple = (max(b) - min(b)) / (max(a) - min(a))
    shift = sum(_to)/len(_to) - sum(_from)/len(_from)
    return multiple, shift


def linear_regression(_from, _to):
    slope, intercept, r, p, std = stats.linregress(_from, _to)
    return slope, intercept


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=3, real_time=False):
    b, a = butter_highpass(cutoff, fs, order=order)
    if real_time == False:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, real_time=False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if real_time == False:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    # y = signal.filtfilt(b, a, data) if real_time is False else signal.lfilter(b,a,data)
    return y


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff_freq, fs, order=3, real_time=False):
    nyq_freq = fs / 2
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    if real_time == False:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


def synchronise_timestamp(imu, holo, show_plot=False):
    """synchronise Hololens <--> imu timestamp with correlating horizontal values

    Args:
        imu (pd.DataFrame): IMU dataframe 
        holo (pd.DataFrame): Hololens dataframe
        show_plot (bool): shows plot of correlation values if True. Defaults to False.

    Returns:
        int,float,float: shift, correlation coef,shifted time(add to imu timestamp)
    """
    time_max = min(holo.timestamp.values[-1], imu.IMUtimestamp.values[-1])
    # holo = holo[holo.timestamp <= time_max]
    imu = imu[imu.IMUtimestamp <= time_max]
    holo_intp = interpolate.interp1d(holo.timestamp, holo.head_rotation_x)
    holo_interpolated = pd.Series(holo_intp(imu.IMUtimestamp))
    approx_range = np.arange(-20, 0)
    rsx = [
        crosscorr(pd.Series(signal.detrend(holo_interpolated)),
                  pd.Series(signal.detrend(imu.rotationX)), lag)
        for lag in approx_range
    ]
    shift = approx_range[np.argmax(rsx)]
    coef = rsx[int(np.argmax(rsx))]
    shift_time = imu.IMUtimestamp.iloc[-1] - imu.IMUtimestamp.iloc[shift]
    if show_plot:
        _, ax = plt.subplots(figsize=(14, 3))
        ax.plot(approx_range, rsx)
        ax.axvline(shift, color='r', linestyle='--')
        plt.show()

    return shift, coef, shift_time


def check_eye_files():
    subjects = range(301, 317)
    envs = ["W", "U"]
    targets = range(8)
    blocks = range(5)
    # subjects = range(304, 305)
    # envs = ["W"]
    # targets = range(0, 1)
    # blocks = range(3, 4)
    for subject in subjects:
        lowcount = 0
        shortcount = 0
        errcount = 0
        for env, target, block in itertools.product(envs, targets, blocks):
            try:
                # print("-" * 10, target, env, block, subject, "-" * 10)
                eye = read_eye_file(target, env, block, subject)
                check_eye_dataframe(eye)

            except Exception as e:
                # print(e.args)
                if e.args[1] == "short":
                    shortcount = shortcount + 1
                elif e.args[1] == "low":
                    lowcount = lowcount + 1
                else:
                    errcount = errcount + 1
        print(
            f"{subject} -> err: {errcount}\tshort: {shortcount}\tlow: {lowcount}"
        )


def check_holo_files():
    subjects = range(301, 317)
    envs = ["W", "U"]
    targets = range(8)
    blocks = range(5)
    for subject in subjects:
        shortcount = 0
        errcount = 0
        practicecount = 0
        for env, target, block in itertools.product(envs, targets, blocks):
            try:
                holo = bring_hololens_data(target, env, block, subject)
                check_hololens_dataframe(holo, block=block, threshold=4.0)

            except Exception as e:
                print(e.args)
                if e.args[0] == 'practice':
                    practicecount += 1
                elif e.args[0] == 'short':
                    shortcount += 1
                else:
                    errcount += 1
        print(
            f"{subject}--> short: {shortcount}, practice: {practicecount}, error: {errcount}"
        )


def filter_visualise(eye, imu):
    eye = eye[eye.confidence > 0.6]
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scatter(x=imu.IMUtimestamp, y=imu.rotationX, name="IMU-vertical"),
        row=1,
        col=1,
    )
    # fig.add_trace(go.Scatter(x=eye.timestamp,y= eye.theta.rolling(window=20).mean()),row=1,col=2,name='eye-filtered')

    b, a = scipy.signal.butter(3, 0.05)
    filtered = scipy.signal.filtfilt(b, a, eye.theta)
    fig.add_trace(
        go.Scatter(x=eye.timestamp,
                   y=filtered,
                   mode="markers",
                   name="eye-filtered-vertical"),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=eye.timestamp, y=eye.theta, name="eye-raw"),
                  row=2,
                  col=1)
    fig.show()


# draw 3d plot of walking trace
def draw_3d_passage(holo):
    fig = px.scatter_3d(
        holo,
        x="head_position_x",
        z="head_position_y",
        y="head_position_z",
        range_x=[-0.5, 0.5],
        range_z=[-0.5, 0.5],
        range_y=[0, 8],
        width=600,
        height=600,
        color="target_entered",
        opacity=0.5,
    )
    fig.update_traces(marker=dict(size=5))

    fig.show()


# simple comparison of hololens & IMU record
def compare_holo_IMU(holo, imu):
    fig = px.line(holo, x="timestamp", y="head_rotation_x")
    # fig.show()
    fig = px.line(imu, x="IMUtimestamp", y="rotationX")
    fig.show()


def euler_to_vector(_x, _y):
    x = math.cos(_y) * math.sin(_x)
    z = math.cos(_y) * math.cos(_x)
    y = math.sin(_y)
    return [x, y, z]


def holo_to_vector(holo: pd.DataFrame):
    vector = []
    for index, row in holo.iterrows():
        holo_vector = np.array(
            euler_to_vector(row['head_rotation_y'] * math.pi / 180, row['head_rotation_x'] * math.pi / 180))
        vector.append(holo_vector)
    holo['vector'] = vector
    return holo


def imu_to_vector(imu: pd.DataFrame):
    vector_x = []
    vector_y = []
    vector_z = []
    vector = []
    for index, row in imu.iterrows():
        imu_vector = np.array(euler_to_vector(row['rotationZ'] * math.pi / 180, row['rotationX'] * math.pi / 180))
        # imu_vector = imu_vector / np.linalg.norm(imu_vector)
        vector.append(imu_vector)
        vector_x.append(imu_vector[0])
        vector_y.append(imu_vector[1])
        vector_z.append(imu_vector[2])
    imu['vector_x'] = vector_x
    imu['vector_y'] = vector_y
    imu['vector_z'] = vector_z
    imu['vector'] = vector
    return imu


class LowPassFilter(object):
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class OneEuroFilter(object):

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))


def one_euro(_data, freq=120, mincutoff=1, beta=1.0, dcutoff=1.0):
    config = dict(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
    filter = OneEuroFilter(**config)
    f = []
    _data = list(_data)
    for i in range(len(_data)):
        f.append(filter(_data[i]))
    return pd.Series(f)


from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
def apply_online_bandpass(b,a, data,order=2):
    result=[]
    # z = signal.lfilter_zi(b, a)
    z=[0]*2*order
    for i,x in enumerate(np.array(data)):
        res,z = signal.lfilter(b,a,[x],zi=z)
        result.append(res[0])
    return pd.Series(result)

# Several flavors of bandpass FIR filters.

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    nyq = 0.5 * fs
    atten = kaiser_atten(ntaps, width / nyq)
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps

def bandpass_remez(ntaps, lowcut, highcut, fs, width):
    delta = 0.5 * width
    edges = [0, lowcut - delta, lowcut + delta,
             highcut - delta, highcut + delta, 0.5*fs]
    taps = remez(ntaps, edges, [0, 1, 0], Hz=fs)
    return taps

if __name__ == "__main__":
    import random

    timestamp = range(100)
    signal = [math.sin(x) for x in timestamp]
    noise = [random.random() / 5 for x in timestamp]
    # signal + (random.random()-0.5)/5.0
    original = [signal[i] + noise[i] for i in timestamp]
    filtered = one_euro(original, beta=0.99, mincutoff=0.87, dcutoff=1.0)
    plt.plot(timestamp, original)
    plt.plot(timestamp, filtered)
    plt.show()
    # duration = 10.0
    # config = {
    #     'freq': 120,
    #     'mincutoff': 1.0,
    #     'beta': 0.5,
    #     'dcutoff': 1.0
    # }
    # f = OneEuroFilter(**config)
    # timestamp = 0.0
    # while timestamp < duration:
    #     signal = math.sin(timestamp)
    #     noisy = signal + (random.random() - 0.5) / 5.0
    #     filtered = f(noisy, timestamp)
    #     print(f"{timestamp}, {signal}, {noisy}, {filtered}")
    #     timestamp += 1.0 / config['freq']
