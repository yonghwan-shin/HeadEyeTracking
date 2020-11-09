# %%
from Mathematics import change_angle, normalize, LinearRegression, real_time_peak_detection, angle_velocity
from scipy import interpolate
from Butterworth_second_order_filters import butterworth_lowpass, butterworth_highpass
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
# pio.renderers.default = "vscode"

""" Record file columns
Index(['timestamp', 'head_position', 'head_rotation', 'head_forward', 'eye_x',
       'eye_y', 'hp_H', 'hp_V', 'hp_H_eye', 'hp_V_eye', 'multiple_H',
       'multiple_V', 'temp_H', 'temp_V'],
"""

# if __name__ == '__main__':
# %%  reading record file
f = open("221/U1.json")
data = pd.DataFrame(json.load(f)["data"])

data.timestamp = data.timestamp - data.timestamp[0]
for col, item in itertools.product(
        ["head_position", "head_rotation", "head_forward", "one_euro_filtered_vector", "targetPosition"],
        ["x", "y", "z"],
):
    data[col + "_" + item] = data[col].apply(pd.Series)[item]

for col in ["head_rotation_x", "head_rotation_y", "head_rotation_z"]:
    data[col] = data[col].apply(change_angle)


def asSpherical(xyz: list):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.acos(z / r) * 180 / math.pi  # to degrees
    phi = math.atan2(y, x) * 180 / math.pi
    return [r, theta, phi]


thetas = []
phis = []
for index, row in data.iterrows():
    x = row["targetPosition_x"] - row["head_position_x"]
    y = row["targetPosition_y"] - row["head_position_y"]
    z = row["targetPosition_z"] - row["head_position_z"]
    [r, theta, phi] = asSpherical([x, z, y])
    thetas.append(90 - theta)
    phis.append(90 - phi)
data['Theta'] = thetas
data["Phi"] = phis
data['TargetVertical'] = data.head_rotation_x + data.Theta
data['TargetHorizontal'] = data.head_rotation_y - data.Phi


def calculate_veclocity(_from, _to, _time):
    if np.isnan(_from) or np.isnan(_to):
        return 0
    return (_to - _from) / _time


data["head_forward_next"] = data.head_forward.shift(1)
data["head_rotation_y_next"] = data.head_rotation_y.shift(1)
data["head_rotation_x_next"] = data.head_rotation_x.shift(1)
data["time_interval"] = data.timestamp.diff()
data["angle_speed"] = data.apply(
    lambda x: angle_velocity(
        x.head_forward, x.head_forward_next, x.time_interval),
    axis=1,
)
data["angle_speed_H"] = data.apply(
    lambda x: calculate_veclocity(
        x.head_rotation_y, x.head_rotation_y_next, x.time_interval),
    axis=1
)
data["angle_speed_V"] = data.apply(
    lambda x: calculate_veclocity(
        x.head_rotation_x, x.head_rotation_x_next, x.time_interval),
    axis=1
)
# data = data[10:]
# %%
# lag = 30
# threshold = 5.0
# influence = 0.05
# testdata = np.array(data.eye_y)
# peak_detect = real_time_peak_detection([], lag, threshold, influence)
# for i in range(lag, len(testdata)):
#     peak_detect.thresholding_algo(testdata[i])
# fig = make_subplots(rows=1, cols=1)


one = one_euro(data.temp_H, data.timestamp, freq=60, mincutoff=0.5, beta=0.1)
origin = one_euro(data.lp_H, data.timestamp,
                  freq=60, mincutoff=0.5, beta=0.1)

fig = go.Figure(
    data=[
        # go.Scatter(x=data.timestamp,y=(data.original_eye_x-data.original_eye_x.mean())*250,name='originaleye'),

        # go.Scatter(x=data.timestamp,y=data.hp_H_eye*250,name='bp-eye'),
        # go.Scatter(x=data.timestamp,y=data.hp_H,name='bp-head')
        # go.Scatter(x=data.timestamp, y=data.head_rotation_y, visible='legendonly'),
        go.Scatter(x=data.timestamp, y=data.lp_H, name='lp-head'),
        # .Scatter(x=data.timestamp,y=data.lp_H_shift,name='lp'),
        # go.Scatter(x=data.timestamp,y=data.confidence,name='confidence'),
        go.Scatter(x=data.timestamp, y=data.hp_H, name='head-hp'),
        # go.Scatter(x=data.timestamp, y=data.lp_H - data.hp_H, name='diff'),
        go.Scatter(x=data.timestamp, y=data.lp_H - data.Phi + data.hp_H_eye * data.current_multiple_H, name='comp'),
        go.Scatter(x=data.timestamp, y=data.hp_H_eye * data.current_multiple_H, name='eye-multiple'),
        go.Scatter(x=data.timestamp, y=data.hp_H_eye * data.multiple_H, name='eye-multiple-non-corrected'),
        go.Scatter(x=data.timestamp, y=data.lp_H - data.Phi + data.hp_H_eye * data.multiple_H, name='comp-non-corrected'),
        go.Scatter(x=data.timestamp, y=data.temp_H, name='record-final', visible='legendonly'),
        go.Scatter(x=data.timestamp, y=data.one_euro_filtered_vector_y, name='one-final', visible='legendonly'),
        go.Scatter(x=data.timestamp, y=data.lp_H - data.Phi, name='target'),
        go.Scatter(x=data.timestamp, y=data.Phi, name='Phi'),
        go.Scatter(x=data.timestamp, y=data.temp_H - data.lp_H, name='minus'),
        # go.Scatter(x=data.timestamp,y=data.multiple_H,name='multiple'),
        # go.Scatter(x=data.timestamp,y=data.current_multiple_H,name='currentMultiple')
        # go.Scatter(x=data.timestamp,y=one,name='one'),
        go.Scatter(x=data.timestamp, y=origin, name='origin'),
    ]
)

# ACCURACY : mean difference
accuracy_lowpass = data.Phi.mean() - data.lp_H.mean()
accuracy_algorithm = data.Phi.mean() - data.one_euro_filtered_vector_y.mean()
# PRECISION : standard deviation
precision_lowpass = data.lp_H.std();
precision_algorithm = data.one_euro_filtered_vector_y.std()
print('accuracy:', accuracy_lowpass, '->', accuracy_algorithm)
print('precision:', precision_lowpass, '->', precision_algorithm)
fig.update_layout(title='Horizontal:' + str(accuracy_lowpass) + '->' + str(accuracy_algorithm) + 'precision ' + str(precision_lowpass) + '->' + str(
    precision_algorithm))
fig.show()

# one_euro(data.temp_H,data.timestamp,freq=60,mincutoff=0.8,beta=1.0)
# %%
# rsq,intercept,slope = LinearRegression(list(data.timestamp),list(data.lp_H))
w = 1
x = list()
y = list()
d = list()
f = list()
output = list()
window = 120
for i in range(1, len(data.timestamp)):
    x.append(data.timestamp.iloc[i])
    y.append(data.lp_H.iloc[i])
    rsq, intercept, slope = LinearRegression(x, y)
    if i > window:
        rsq, intercept, slope = LinearRegression(x[-window:], y[-window:])

    f.append(slope * x[-1] + intercept)

    d.append(abs(y[-1] - f[-1]))
    output.append(y[-1] - f[-1])
    if i == 1:
        continue
    if d[-1] / np.std(d) > 5:
        x.pop();
        y.pop();
        f.pop();
        d.pop();
# %%
fig = go.Figure(
    data=[

        go.Scatter(x=data.timestamp, y=y, name='y'),
        go.Scatter(x=data.timestamp, y=f, name='f'),
        go.Scatter(x=data.timestamp, y=output, name='output'),
        go.Scatter(x=data.timestamp, y=data.hp_H, name='highpass'),
        go.Scatter(x=data.timestamp, y=data.TargetHorizontal, name='Target')
    ]
)
fig.update_layout(title='Horizontal')
fig.show()
# %% distribution plot
fig = (ff.create_distplot(hist_data=[data.head_rotation_y[900:1320],
                                     data.temp_H[900:1320]], group_labels=['head', 'final'], bin_size=0.1))
fig.show()
# %%
data["hp_V_shift"] = data.hp_V.shift(3)
highpass_V = [0, 0]
for i in range(2, len(data.head_rotation_x)):
    highpass_V.append(butterworth_highpass(0.20, data.timestamp.iloc[i] - data.timestamp.iloc[i - 1],
                                           data.head_rotation_x.iloc[i], data.head_rotation_x.iloc[i - 1], data.head_rotation_x.iloc[i - 2],
                                           highpass_V[i - 1], highpass_V[i - 2]))
# highpass_V = [0]*7+highpass_V
fig = go.Figure(
    data=[
        # go.Scatter(x=data.timestamp, y=data.head_rotation_x,
        #            name='head', visible='legendonly'),
        go.Scatter(x=data.timestamp, y=data.lp_V, name='head'),
        # go.Scatter(x=data.timestamp, y=(data.eye_y-data.eye_y.mean())*20,name='eye-original'),
        go.Scatter(x=data.timestamp, y=data.hp_V, name='head-hp'),
        # go.Scatter(x=data.timestamp, y=highpass_V, name='head-hp-0.1'),
        # go.Scatter(x=data.timestamp, y=data.head_rotation_x - highpass_V, name='0.1'),
        go.Scatter(x=data.timestamp, y=data.lp_V - data.hp_V, name='diff'),
        go.Scatter(x=data.timestamp, y=data.lp_V + data.Theta + \
                                       data.hp_V_eye * data.current_multiple_V, name='comp'),
        go.Scatter(x=data.timestamp, y=data.hp_V_eye * \
                                       data.multiple_V, name='eye-multiple'),
        go.Scatter(x=data.timestamp, y=data.lp_V + data.Theta, name='target'),
        go.Scatter(x=data.timestamp, y=-data.Theta, name='theta'),
        # go.Scatter(x=data.timestamp,y=data.hp_V_eye*200,name='eye-200'),
        go.Scatter(x=data.timestamp, y=data.temp_V,
                   name='recorded_final', visible='legendonly'),
        go.Scatter(x=data.timestamp, y=data.one_euro_filtered_vector_x, name='one-final', visible='legendonly'),
        # go.Scatter(x=data.timestamp,y=highpass_h,name='test-head-hp')
    ]
)
# ACCURACY : mean difference
accuracy_lowpass = data.Theta.mean() - data.lp_V.mean()
accuracy_algorithm = data.Theta.mean() - data.one_euro_filtered_vector_x.mean()
# PRECISION : standard deviation
precision_lowpass = data.lp_V.std();
precision_algorithm = data.one_euro_filtered_vector_x.std()
print('accuracy:', accuracy_lowpass, '->', accuracy_algorithm)
print('precision:', precision_lowpass, '->', precision_algorithm)
fig.update_layout(title='Vertical:' + str(accuracy_lowpass) + '->' + str(accuracy_algorithm) + 'precision ' + str(precision_lowpass) + '->' + str(
    precision_algorithm))
fig.show()

#%%
fig= go.Figure(
    data=[
        go.Scatter(x=data.timestamp,y=data.original_eye_x,name='original-eye-x'),
        go.Scatter(x=data.timestamp,y=data.eye_x,name='eye-x'),
        go.Scatter(x=data.timestamp,y=data.lp_H_eye,name='eye_lowpass'),
        go.Scatter(x=data.timestamp,y=data.hp_H_eye,name='eye_highpass'),
        go.Scatter(x=data.timestamp,y=data.lp_H_eye - data.hp_H_eye,name='eye_diff'),
    ]
)
fig.show()
# %%

total_time = np.arange(0, 30, 1 / 60)
head_intp_v = interpolate.interp1d(
    data.timestamp, data.head_rotation_x, fill_value='extrapolate')
eye_intp_v = interpolate.interp1d(
    data.timestamp, data.eye_y, fill_value='extrapolate')

head_intp_h = interpolate.interp1d(
    data.timestamp, data.head_rotation_y, fill_value='extrapolate')
eye_intp_h = interpolate.interp1d(
    data.timestamp, data.eye_x, fill_value='extrapolate')
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

diff_vs = []
count = 0
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
    out_h = (head_intp_h(t + window) +
             (-_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h) / 2
    out_v = (head_intp_v(t + window) - (_S_v * (eye_intp_v(t +
                                                           window) - _E_v.mean())) + _I_v + _S_v * _E_v.mean()) / 2
    # out_h = (head_intp_h(t + window) - (_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h + _S_h * _E_h.mean()) / 2

    final_v.append(out_v)
    final_h.append(out_h)

    diff_v = - (_S_v * (eye_intp_v(t + window) - _E_v.mean())) + \
             _I_v + _S_v * _E_v.mean()
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
