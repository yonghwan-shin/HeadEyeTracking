# This is a sample Python script.
import pandas as pd
import json
import itertools
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.renderers.default = "browser"
import math


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Index(['timestamp', 'head_position', 'head_rotation', 'head_forward', 'eye_x',
#        'eye_y', 'hp_H', 'hp_V', 'hp_H_eye', 'hp_V_eye', 'multiple_H',
#        'multiple_V', 'temp_H', 'temp_V'],

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


def one_euro(_data, timestamp=None, freq=120, mincutoff=1, beta=1.0, dcutoff=1.0):
    config = dict(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
    filter = OneEuroFilter(**config)
    f = []
    _data = list(_data)

    for i in range(len(_data)):
        if timestamp is None:
            f.append(filter(x=_data[i]))
        else:
            f.append(filter(_data[i], timestamp=timestamp[i]))
    return pd.Series(f)


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #%%
    f = open("test_8.json")
    data = pd.DataFrame(json.load(f)["data"])
    data.timestamp = data.timestamp - data.timestamp[0]
    for col, item in itertools.product(
            ["head_position", "head_rotation", "head_forward"],
            ["x", "y", "z"],
    ):
        data[col + "_" + item] = data[col].apply(pd.Series)[item]

    for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
        data[col] = data[col].apply(change_angle)
    # %%
    eye_hp = data.eye_y - one_euro(data.eye_y, data.timestamp, 60, 0.1, 0.001, 1.0)
    head_hp = data.head_rotation_x - one_euro(data.head_rotation_x, data.timestamp, 60, 0.1, 0.001, 1.0)
    fig = go.Figure(
        data=[
            go.Scatter(x=data.timestamp, y=data.head_rotation_x, name='V-head'),
            go.Scatter(x=data.timestamp, y=data.hp_V_eye * data.multiple_V, name='hp-eye-multiple'),
            go.Scatter(x=data.timestamp, y=data.hp_V_eye * 200, name='hp-eye-200'),

            go.Scatter(x=data.timestamp, y=data.hp_V, name='hp-head'),
            go.Scatter(x=data.timestamp, y=data.temp_V, name='temp'),
            # go.Scatter(x=data.timestamp, y=one_euro(data.head_rotation_x, data.timestamp, 60, 0.1, 0.001, 1.0), name='hp-diff')
        ]
    )
    fig.show()

    fig = go.Figure(
        data=[
            go.Scatter(x=data.timestamp, y=data.eye_y, name='V-eye'),
            go.Scatter(x=data.timestamp, y=data.hp_V_eye, name='hp-Veye'),
            go.Scatter(x=data.timestamp, y=eye_hp, name='hp-test'),
            # go.Scatter(x=data.timestamp, y=data.eye_y - one_euro(data.eye_y, data.timestamp, 60, 0.1, 0.1, 0.5), name='hp-test'),
        ]
    )
    fig.show()

    # %%
    eye_hp = data.eye_x - one_euro(data.eye_x, data.timestamp, 60, 0.1, 0.001, 1.0)
    head_hp = data.head_rotation_y - one_euro(data.head_rotation_y, data.timestamp, 60, 0.1, 0.001, 1.0)
    # head_hp = one_euro(data.head_rotation_y, data.timestamp, 60, 0.1, 0.001, 1.0)
    fig = go.Figure(
        data=[
            go.Scatter(x=data.timestamp, y=data.head_rotation_y, name='H-head'),
            go.Scatter(x=data.timestamp, y=data.hp_H_eye * data.multiple_H, name='hp-eye-multiple'),
            # go.Scatter(x=data.timestamp, y=data.hp_H_eye * 100, name='hp-eye-100'),
            go.Scatter(x=data.timestamp, y=(data.eye_x - data.eye_x.mean()) * 100, name='hp-eye-100'),
            go.Scatter(x=data.timestamp, y=data.hp_H, name='hp-head'),

            go.Scatter(x=data.timestamp, y=data.temp_H, name='temp'),

        ]
    )
    fig.show()

    fig = go.Figure(
        data=[
            go.Scatter(x=data.timestamp, y=data.eye_x, name='H-eye'),
            go.Scatter(x=data.timestamp, y=data.hp_H_eye, name='hp-Heye'),
            go.Scatter(x=data.timestamp, y=eye_hp, name='hp-test'),
            go.Scatter(x=data.timestamp,y=one_euro(data.eye_x, data.timestamp, 60, 0.1, 0.001, 1.0),name='diff')
        ]
    )
    fig.show()
# %%

plt.plot(data.multiple_V)
plt.show()
