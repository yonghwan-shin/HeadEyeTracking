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

from filehandling import (bring_data, bring_hololens_data, check_eye_dataframe,
                          check_hololens_dataframe, read_eye_file)


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
        ax.plot(approx_range, rs)
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


if __name__ == "__main__":
    t = pd.Series([math.sin(a) for a in range(1, 100)])
    t2 = pd.Series([math.sin(a) for a in range(5, 104)])
    rs = [crosscorr(t, t2, lag) for lag in range(-5, 5)]
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.argmax(rs),
               color='r',
               linestyle='--',
               label='peak synchrony')
    plt.legend()
    plt.show()
