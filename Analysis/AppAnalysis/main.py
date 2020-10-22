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
def winter_low(cutoff_freq, sample_time, x0, x1, x2, y1, y2, print_coeff=False):
    """Filters a data sample based on two past unfiltered and filtered data samples.

    2nd order low pass, single pass butterworth filter presented in Winter2009.

    Parameters
    ==========
    cuttoff_freq: float
        The desired lowpass cutoff frequency in Hertz.
    sample_time: floaat
        The difference in time between the current time and the previous time.
    x0 : float
        The current unfiltered signal, x_i
    x1 : float
        The unfiltered signal at the previous sampling time, x_i-1.
    x2 : float
        The unfiltered signal at the second previous sampling time, x_i-2.
    y1 : float
        The filtered signal at the previous sampling time, y_i-1.
    y2 : float
        The filtered signal at the second previous sampling time, y_i-2.

    """
    sampling_rate = 1 / sample_time  # Hertz

    correction_factor = 1.0  # 1.0 for a single pass filter

    corrected_cutoff_freq = np.tan(np.pi * cutoff_freq / sampling_rate) / correction_factor  # radians

    K1 = np.sqrt(2) * corrected_cutoff_freq
    K2 = corrected_cutoff_freq ** 2

    a0 = K2 / (1 + K1 + K2)
    a1 = 2 * a0
    a2 = a0

    K3 = a1 / K2

    b1 = -a1 + K3
    b2 = 1 - a1 - K3

    if print_coeff:
        print('num:', a0, a1, a2)
        print('dem:', 1.0, -b1, -b2)

    return a0 * x0 + a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2


def murphy_high(cutoff_freq, sample_time, x0, x1, x2, y1, y2, print_coeff=False):
    """
    Parameters
    ==========
    cuttoff_freq: float
        The desired lowpass cutoff frequency in Hertz.
    sample_time: floaat
        The difference in time between the current time and the previous time.
    x0 : float
        The current unfiltered signal, x_i
    x1 : float
        The unfiltered signal at the previous sampling time, x_i-1.
    x2 : float
        The unfiltered signal at the second previous sampling time, x_i-2.
    y1 : float
        The filtered signal at the previous sampling time, y_i-1.
    y2 : float
        The filtered signal at the second previous sampling time, y_i-2.
    """
    sampling_rate = 1 / sample_time  # Hertz

    correction_factor = 1.0

    cutoff_freq = 1 / 2 / sample_time - cutoff_freq  # covert high pass freq to equivalent lowpass freq

    corrected_cutoff_freq = np.tan(np.pi * cutoff_freq / sampling_rate) / correction_factor

    K1 = np.sqrt(2) * corrected_cutoff_freq
    K2 = corrected_cutoff_freq ** 2

    a0 = K2 / (1 + K1 + K2)
    a1 = 2 * a0
    a2 = a0

    K3 = a1 / K2

    b1 = -a1 + K3
    b2 = 1 - a1 - K3

    c0 = a0
    c1 = -a1
    c2 = a2

    d1 = -b1
    d2 = b2

    if print_coeff:
        print('num:', c0, c1, c2)
        print('dem:', 1.0, -d1, -d2)

    return c0 * x0 + c1 * x1 + c2 * x2 + d1 * y1 + d2 * y2


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


def LinearRegression(x, y):
    if len(x) != len(y):
        raise Exception("different length of x, y")
    sumOfX = 0
    sumOfY = 0
    sumOfXSq = 0
    sumOfYSq = 0
    sumCodeviates = 0
    for i in range(len(x)):
        sumCodeviates += x[i] * y[i]
        sumOfX += x[i]
        sumOfY += y[i]
        sumOfXSq += x[i] * x[i]
        sumOfYSq += y[i] * y[i]
    count = len(x)
    ssX = sumOfXSq - ((sumOfX * sumOfX) / count)
    ssY = sumOfYSq - ((sumOfY * sumOfY) / count)
    rNumerator = (count * sumCodeviates) - (sumOfX * sumOfY)
    rDenom = (count * sumOfXSq - (sumOfX * sumOfX)) * (count * sumOfYSq - (sumOfY * sumOfY))
    sCo = sumCodeviates - ((sumOfX * sumOfY) / count)
    meanX = sumOfX / count
    meanY = sumOfY / count
    dblR = rNumerator / math.sqrt(rDenom)
    rSquared = dblR * dblR
    yIntercept = meanY - ((sCo / ssX) * meanX)
    slope = sCo / ssX
    return rSquared, yIntercept, slope


def normalize(_from, _to, RMS=False):
    if RMS == True:
        _from = np.array(_from)
        _to = np.array(_to)
        mean_difference = _to.mean() - _from.mean()
        multiple_power2 = np.power((_to - _to.mean()), 2).mean() / np.power((_from - _from.mean()), 2).mean()
        multiple = np.sqrt(multiple_power2)
        return multiple, mean_difference
    a = _from - sum(_from) / len(_from)
    b = _to - sum(_to) / len(_to)
    multiple = (max(b) - min(b)) / (max(a) - min(a))
    shift = sum(_to) / len(_to) - sum(_from) / len(_from)
    return multiple, shift


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
# if __name__ == '__main__':
# %%
# eye = pd.read_csv('log_10.csv')
# eye.timestamp = eye.timestamp - eye.timestamp[0]
import numpy as np

f = open("Pilot4.json")
data = pd.DataFrame(json.load(f)["data"])
# data = pd.DataFrame(json.load(f))
data.timestamp = data.timestamp - data.timestamp[0]
for col, item in itertools.product(
        ["head_position", "head_rotation", "head_forward"],
        ["x", "y", "z"],
):
    data[col + "_" + item] = data[col].apply(pd.Series)[item]

for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
    data[col] = data[col].apply(change_angle)


# %%
class real_time_peak_detection():
    def __init__(self, array, lag, threshold, influence):
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        # self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        # self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
            return 0

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]

            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i]


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


lag = 30
threshold = 5.0
influence = 0.05
testdata = np.array(data.eye_y)
peak_detect = real_time_peak_detection([], lag, threshold, influence)
# peaks=[0]*10
# signals=[]
# filtered_ys = []
# avgFilters=[]
# stdFilters=[]

for i in range(lag, len(testdata)):
    peak_detect.thresholding_algo(testdata[i])

# threshold = 3.5
# testdata =  np.array(data.eye_y[300:])
# result = thresholding_algo(testdata, 5, threshold, 0.30)

fig = go.Figure(
    data=[
        go.Scatter(x=data.timestamp, y=data.head_rotation_y,name='head'),
        # go.Scatter(x=data.timestamp,y=data.confidence,name='confidence'),
        # go.Scatter(x=data.timestamp[300:], y=result['avgFilter'], name='peaks'),
        # go.Scatter(x=data.timestamp[300:], y=result['avgFilter'] + threshold * result['stdFilter'], name='peaks++'),
        # go.Scatter(x=data.timestamp[300:], y=result['avgFilter'] - threshold * result['stdFilter'], name='peaks--'),
        # go.Scatter(x=data.timestamp[300:], y=result['signals'], name='signal'),
        # go.Scatter(x=data.timestamp[300:], y=(data.eye_x)[300:] , name='eye-original'),
        # go.Scatter(x=data.timestamp, y=data.hp_H_eye*200,name='eye-hp'),
        # go.Scatter(x=data.timestamp[lag:], y=data.eye_x[lag:] , name='eye-original'),
        # go.Scatter(x=data.timestamp, y=peak_detect.y, name='y'),
        # go.Scatter(x=data.timestamp, y=peak_detect.signals, name='sig'),
        # go.Scatter(x=data.timestamp, y=peak_detect.filteredY, name='fy'),
        # go.Scatter(x=data.timestamp, y=pd.Series(peak_detect.avgFilter) + peak_detect.threshold * pd.Series(peak_detect.stdFilter), name='++'),
        # go.Scatter(x=data.timestamp, y=pd.Series(peak_detect.avgFilter) - peak_detect.threshold * pd.Series(peak_detect.stdFilter), name='--'),
        go.Scatter(x=data.timestamp, y=data.hp_H, name='head-hp'),
        go.Scatter(x=data.timestamp, y=data.hp_H + data.hp_H_eye * data.multiple_H, name='comp'),
        go.Scatter(x=data.timestamp, y=data.hp_H_eye * data.multiple_H, name='eye-multiple'),
        go.Scatter(x=data.timestamp,y=data.temp_H,name='record-final')
        #
        # go.Scatter(x=data.timestamp,y=data.hp_H,name='head-hp'),

    ]
)
fig.show()
# %%
fig = go.Figure(
    data=[
        go.Scatter(x=data.timestamp, y=data.head_rotation_x,name='head'),
        # go.Scatter(x=data.timestamp, y=data.eye_x*200,name='eye-original'),
        go.Scatter(x=data.timestamp, y=data.hp_V, name='head-hp'),
        go.Scatter(x=data.timestamp, y=data.hp_V + data.hp_V_eye * data.multiple_V, name='comp'),
        go.Scatter(x=data.timestamp, y=data.hp_V_eye * data.multiple_V, name='eye-multiple'),
        # go.Scatter(x=data.timestamp,y=data.hp_V_eye*200,name='eye-200'),
        go.Scatter(x=data.timestamp,y=data.temp_V,name='recorded_final')
        # go.Scatter(x=data.timestamp,y=highpass_h,name='test-head-hp')
    ]
)
fig.show()
# %%
#     # %%
#
#     eye_hp = data.eye_y - one_euro(data.eye_y, data.timestamp, 60, 0.1, 0.001, 1.0)
#     head_hp = data.head_rotation_x - one_euro(data.head_rotation_x, data.timestamp, 60, 0.1, 0.001, 1.0)
#     fig = go.Figure(
#         data=[
#             go.Scatter(x=data.timestamp, y=data.head_rotation_x, name='V-head'),
#
#             go.Scatter(x=data.timestamp , y=data.hp_V_eye * data.multiple_V, name='hp-eye-multiple'),
#             go.Scatter(x=data.timestamp, y=data.hp_V_eye * 200, name='hp-eye-200'),
#
#             go.Scatter(x=data.timestamp, y=data.hp_V, name='hp-head'),
#             go.Scatter(x=data.timestamp, y=data.temp_V, name='temp'),
#
#             # go.Scatter(x=data.timestamp, y=one_euro(data.head_rotation_x, data.timestamp, 60, 0.1, 0.001, 1.0), name='hp-diff')
#         ]
#     )
#     fig.show()
#
#     fig = go.Figure(
#         data=[
#             go.Scatter(x=data.timestamp, y=data.eye_y, name='V-eye'),
#             go.Scatter(x=data.timestamp, y=data.hp_V_eye, name='hp-Veye'),
#             go.Scatter(x=data.timestamp, y=eye_hp, name='hp-test'),
#             # go.Scatter(x=eye.timestamp,y=eye.y,name='bett')
#             # go.Scatter(x=data.timestamp, y=data.eye_y - one_euro(data.eye_y, data.timestamp, 60, 0.1, 0.1, 0.5), name='hp-test'),
#         ]
#     )
#     fig.show()
#
#     # %%
#     eye_hp = data.eye_x - one_euro(data.eye_x, data.timestamp, 60, 0.1, 0.001, 1.0)
#     head_hp = data.head_rotation_y - one_euro(data.head_rotation_y, data.timestamp, 60, 0.1, 0.001, 1.0)
#     shift_test = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
#
#     fig = go.Figure(
#         data=[
#
#             # go.Scatter(x=data.timestamp+ 0.2, y=data.head_rotation_y, name='H-head-shift'),
#             go.Scatter(x=data.timestamp , y=data.head_rotation_y, name='H-head'),
#             go.Scatter(x=data.timestamp, y=data.hp_H_eye * data.multiple_H, name='hp-eye-multiple'),
#             go.Scatter(x=data.timestamp, y=data.hp_H_eye * 200, name='hp-eye-100'),
#             # go.Scatter(x=data.timestamp, y=(data.eye_x - data.eye_x.mean()) * -100, name='hp-eye-100'),
#
#             # go.Scatter(x=eye.timestamp, y=(eye.x-eye.x.mean())*200, name='bett'),
#             go.Scatter(x=data.timestamp, y=data.hp_H, name='hp-head'),
#             go.Scatter(x=data.timestamp,y=data.head_rotation_y + data.hp_H_eye*data.multiple_H,name='est'),
#             go.Scatter(x=data.timestamp, y=one_euro(data.head_rotation_y + data.hp_H_eye * data.multiple_H,data.timestamp,60,3,0.01), name='ff'),
#             # go.Scatter(x=data.timestamp, y=one_euro(data.head_rotation_y,data.timestamp,60,3,0.01), name='H-head-ff'),
#             # go.Scatter(x=data.timestamp, y=data.temp_H, name='temp'),
#
#         ]
#     )
#     fig.show()
#     #%%
#     fig = go.Figure(
#         data=[
#             go.Scatter(x=data.timestamp, y=data.eye_x, name='H-eye'),
#             go.Scatter(x=data.timestamp, y=data.hp_H_eye, name='hp-Heye'),
#             go.Scatter(x=data.timestamp, y=eye_hp, name='hp-test'),
#             # go.Scatter(x=eye.timestamp+2.2, y=eye.x, name='bett'),
#             go.Scatter(x=data.timestamp,y=one_euro(data.eye_x, data.timestamp, 60, 0.5, 0.01, 1.0),name='diff')
#         ]
#     )
#     fig.show()
# # %%
#
# plt.plot(data.multiple_H)
# plt.plot(data.multiple_H.rolling(100,min_periods=1).mean())
# plt.show()
# %%
from scipy import interpolate
import numpy as np

total_time = np.arange(0, 30, 1 / 60)
head_intp_v = interpolate.interp1d(data.timestamp, data.head_rotation_x, fill_value='extrapolate')
eye_intp_v = interpolate.interp1d(data.timestamp, data.eye_y, fill_value='extrapolate')

head_intp_h = interpolate.interp1d(data.timestamp, data.head_rotation_y, fill_value='extrapolate')
eye_intp_h = interpolate.interp1d(data.timestamp, data.eye_x, fill_value='extrapolate')
H_v = head_intp_v(total_time)
E_v = eye_intp_v(total_time)
H_h = head_intp_h(total_time)
E_h = eye_intp_h(total_time)
window = 1.0
squares_v = []
Slope_v = []
Intercept_v = []
final_v = []
squares_h = []
Slope_h = []
Intercept_h = []
final_h = []

diff_vs = [];
count = 0;
diff_hs = []
for t in total_time:
    if t >= 30 - window:
        break
    timeline = np.arange(t, t + window, 1 / 60)
    _H_v = head_intp_v(timeline)
    _E_v = eye_intp_v(timeline)
    _H_h = head_intp_h(timeline)
    _E_h = eye_intp_h(timeline)
    rsq_v, itcp_v, slope_v = LinearRegression(_E_v, _H_v)
    # rsq_h, itcp_h, slope_h = LinearRegression(_E_h-_E_h.mean(), _H_h)
    slope_h, itcp_h = normalize(_E_h, _H_h, RMS=True)

    slope_h = -slope_h

    # squares_v.append(rsq_v)
    Slope_v.append(slope_v)
    Intercept_v.append(itcp_v)
    # squares_h.append(rsq_h)
    Slope_h.append(slope_h)
    Intercept_h.append(itcp_h)
    # _S_v = sum(Slope_v) / len(Slope_v)
    # _I_v = sum(Intercept_v) / len(Intercept_v)
    # _S_h = sum(Slope_h) / len(Slope_h)
    # _I_h = sum(Intercept_h) / len(Intercept_h)
    # temp_h = [sh for sh in Slope_h if sh < 0]

    _S_v = slope_v
    _S_h = slope_h
    _I_v = itcp_v
    _I_h = itcp_h
    # _S_v = sum(Slope_v[int(-watching_period * window):]) / len(Slope_v[int(-watching_period * window):])
    # _I_v = sum(Intercept_v[int(-watching_period * window):]) / len(Intercept_v[int(-watching_period * window):])
    # _S_h = sum(Slope_h[int(-watching_period * window):]) / len(Slope_h[int(-watching_period * window):])
    # _I_h = sum(Intercept_h[int(-watching_period * window):]) / len(Intercept_h[int(-watching_period * window):])
    # main calculation
    # out_v = (head_intp_v(t + window) - (_S_v * (eye_intp_v(t + window))) + _I_v) / 2
    out_h = (head_intp_h(t + window) + (-_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h) / 2
    out_v = (head_intp_v(t + window) - (_S_v * (eye_intp_v(t + window) - _E_v.mean())) + _I_v + _S_v * _E_v.mean()) / 2
    # out_h = (head_intp_h(t + window) - (_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h + _S_h * _E_h.mean()) / 2

    final_v.append(out_v)
    final_h.append(out_h)

    diff_v = - (_S_v * (eye_intp_v(t + window) - _E_v.mean())) + _I_v + _S_v * _E_v.mean()
    # diff_h = + (_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h + _S_h * _E_h.mean()
    diff_h = (-_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h
    diff_vs.append(diff_v)
    diff_hs.append(diff_h)

plt.plot(total_time, H_h, 'r-')
plt.plot(total_time[int(60 * window) + count:], final_h, 'b-')
plt.plot(total_time[int(60 * window) + count:], diff_hs, 'y-')
plt.title(str(window) + " H-> " + str(sum(Slope_h) / len(Slope_h)))

# plt.plot(total_time,H-phi_intp(total_time)) # Target
plt.grid()
plt.show()

plt.plot(total_time, H_v, 'r-')
plt.plot(total_time[int(60 * window):], final_v, 'b-')
plt.plot(total_time[int(60 * window):], diff_vs, 'y-')
plt.title(str(sum(Slope_v) / len(Slope_v)) + "V")

# plt.plot(total_time,H-phi_intp(total_time)) # Target
plt.grid()
plt.show()
