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
    slope, intercept, r, p, std = stats.linregress(_from, _to)
    return slope, intercept


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
        holo_vector = np.array(euler_to_vector(row['head_rotation_y'] * math.pi / 180, row['head_rotation_x'] * math.pi / 180))
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
        if alpha<=0 or alpha>1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
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

def one_euro(_data,freq=120,mincutoff=1,beta=1.0,dcutoff=1.0):
    config=dict(freq=freq,mincutoff=mincutoff,beta=beta,dcutoff=dcutoff)
    filter=OneEuroFilter(**config)
    f=[]
    for i in range(len(_data)):
        f.append(filter(_data[i]))
    return pd.Series(f)

if __name__ == "__main__":
    import random
    timestamp = range(100)
    signal = [math.sin(x) for x in timestamp]
    noise =[random.random()/5 for x in timestamp]
        # signal + (random.random()-0.5)/5.0
    original = [signal[i]+noise[i] for i in timestamp]
    filtered = one_euro(original,beta=0.99,mincutoff=0.87,dcutoff=1.0)
    plt.plot(timestamp,original)
    plt.plot(timestamp,filtered)
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
