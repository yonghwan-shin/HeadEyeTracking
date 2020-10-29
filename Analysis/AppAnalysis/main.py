import pandas as pd
import json
import itertools
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np
import math
from OneEuroFilter import *

pio.renderers.default = "browser"

from Butterworth_second_order_filters import butterworth_lowpass, butterworth_highpass
from Mathematics import change_angle, normalize, LinearRegression, real_time_peak_detection,angle_velocity

""" Record file columns
Index(['timestamp', 'head_position', 'head_rotation', 'head_forward', 'eye_x',
       'eye_y', 'hp_H', 'hp_V', 'hp_H_eye', 'hp_V_eye', 'multiple_H',
       'multiple_V', 'temp_H', 'temp_V'],
"""

# if __name__ == '__main__':
# %%  reading record file
f = open("800/4.json")
data = pd.DataFrame(json.load(f)["data"])

data.timestamp = data.timestamp - data.timestamp[0]
for col, item in itertools.product(
        ["head_position", "head_rotation", "head_forward"],
        ["x", "y", "z"],
):
    data[col + "_" + item] = data[col].apply(pd.Series)[item]

for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
    data[col] = data[col].apply(change_angle)
def calculate_veclocity(_from,_to,_time):
    if np.isnan(_from) or np.isnan(_to):
        return 0
    return (_to-_from)/_time
data["head_forward_next"] = data.head_forward.shift(1)
data["head_rotation_y_next"] = data.head_rotation_y.shift(1)
data["head_rotation_x_next"] = data.head_rotation_x.shift(1)
data["time_interval"] = data.timestamp.diff()
data["angle_speed"] = data.apply(
    lambda x: angle_velocity(x.head_forward, x.head_forward_next, x.time_interval),
    axis=1,
)
data["angle_speed_H"] = data.apply(
    lambda x: calculate_veclocity(x.head_rotation_y,x.head_rotation_y_next,x.time_interval),
    axis=1
)
data["angle_speed_V"] = data.apply(
    lambda x: calculate_veclocity(x.head_rotation_x,x.head_rotation_x_next,x.time_interval),
    axis=1
)
data = data[10:]
# %%
# lag = 30
# threshold = 5.0
# influence = 0.05
# testdata = np.array(data.eye_y)
# peak_detect = real_time_peak_detection([], lag, threshold, influence)
# for i in range(lag, len(testdata)):
#     peak_detect.thresholding_algo(testdata[i])
# fig = make_subplots(rows=1, cols=1)
one=one_euro(data.temp_H,data.timestamp,freq=60,mincutoff=0.5,beta=0.1)
origin = one_euro(data.head_rotation_y,data.timestamp,freq=60,mincutoff=0.5,beta=0.1)
fig=go.Figure(
    data=[
       # go.Scatter(x=data.timestamp,y=(data.original_eye_x-data.original_eye_x.mean())*250,name='originaleye'),

        # go.Scatter(x=data.timestamp,y=data.hp_H_eye*250,name='bp-eye'),
        # go.Scatter(x=data.timestamp,y=data.hp_H,name='bp-head')
        go.Scatter(x=data.timestamp, y=data.head_rotation_y, name='head', visible='legendonly'),
        # go.Scatter(x=data.timestamp,y=data.confidence,name='confidence'),
        go.Scatter(x=data.timestamp, y=data.hp_H, name='head-hp'),
        go.Scatter(x=data.timestamp, y=data.hp_H + data.hp_H_eye * data.multiple_H, name='comp'),
        go.Scatter(x=data.timestamp, y=data.hp_H_eye * data.multiple_H, name='eye-multiple'),
        go.Scatter(x=data.timestamp, y=data.temp_H, name='record-final', visible='legendonly'),
       # go.Scatter(x=data.timestamp,y=data.multiple_H,name='multiple'),
        #go.Scatter(x=data.timestamp,y=data.current_multiple_H,name='currentMultiple')
        # go.Scatter(x=data.timestamp,y=one,name='one'),
        # go.Scatter(x=data.timestamp,y=origin,name='origin'),
    ]
)
# one_euro(data.temp_H,data.timestamp,freq=60,mincutoff=0.8,beta=1.0)

fig.update_layout(title='Horizontal')
fig.show()
#%% distribution plot
fig=(ff.create_distplot(hist_data=[data.head_rotation_y[900:1320],data.temp_H[900:1320]],group_labels=['head','final'],bin_size=0.1))
fig.show()
# %%
fig = go.Figure(
    data=[
       go.Scatter(x=data.timestamp, y=data.head_rotation_x, name='head', visible='legendonly'),
        #go.Scatter(x=data.timestamp, y=(data.eye_y-data.eye_y.mean())*20,name='eye-original'),
        go.Scatter(x=data.timestamp, y=data.hp_V, name='head-hp'),
         go.Scatter(x=data.timestamp, y=data.hp_V + data.hp_V_eye * data.multiple_V, name='comp'),
        go.Scatter(x=data.timestamp, y=data.hp_V_eye * data.multiple_V, name='eye-multiple'),
        # go.Scatter(x=data.timestamp,y=data.hp_V_eye*200,name='eye-200'),
        go.Scatter(x=data.timestamp, y=data.temp_V, name='recorded_final', visible='legendonly')
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
