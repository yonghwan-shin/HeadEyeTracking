"""
NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi

holo.columns
Index(['timestamp', 'head_position', 'head_rotation', 'head_forward',
       'target_position', 'target_entered', 'angular_distance',
       'head_position_x', 'head_position_y', 'head_position_z',
       'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
       'head_forward_x', 'head_forward_y', 'head_forward_z',
       'target_position_x', 'target_position_y', 'target_position_z'],
      dtype='object')

imu.columns
Index(['IMUtimestamp', 'rotationX', 'rotationY', 'rotationZ'], dtype='object')

eye.columns
Index(['circle_3d', 'confidence', 'timestamp', 'diameter_3d', 'ellipse',
       'location', 'diameter', 'sphere', 'projected_sphere',
       'model_confidence', 'model_id', 'model_birth_timestamp', 'theta', 'phi',
       'norm_pos', 'topic', 'id', 'method', 'python_timestamp', 'norm_x',
       'norm_y'],
      dtype='object')
"""

# %%bandwidth test

from plotly.subplots import make_subplots
from analysing_functions import *
from IIRfilter import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal, stats

pio.renderers.default = "browser"
target = 3

env = "W"
block = 2
subject = 301

holo, imu, eye = bring_data(target, env, block, subject)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)

# eye = eye[eye.confidence > 0.8]
eye.timestamp = eye.timestamp - shift_time
eye.norm_x = eye.norm_x - eye.norm_x[0]
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)

offset = 240
#
"""
horizontal way : -> abandoned
iirfilter on eye-norm-x : order=2, lowcut = 0.001, highcut = 0.05, bandpass
iirfilter on head-rotation-y : order=2, lowcut = 0.001, highcut = 0.05, bandpass
head and eye have opposite signal
apply on lowpass- head : order=2,lowcut=0.05
"""
head_h = one_euro(timestamp=holo.timestamp, _data=holo.head_rotation_y, freq=60, mincutoff=1, beta=0.1, dcutoff=1.0)
eye_h = one_euro(timestamp=eye.timestamp, _data=eye.norm_x, freq=120, mincutoff=2, beta=-.1, dcutoff=1.0)
head_v = one_euro(timestamp=holo.timestamp, _data=holo.head_rotation_x, freq=60, mincutoff=1, beta=0.1, dcutoff=1.0)
eye_v = one_euro(timestamp=eye.timestamp, _data=eye.norm_y, freq=120, mincutoff=2, beta=-.1, dcutoff=1.0)

head_intp_v = interpolate.interp1d(holo.timestamp, head_v, fill_value='extrapolate')
eye_intp_v = interpolate.interp1d(eye.timestamp, eye_v, fill_value='extrapolate')
theta_intp = interpolate.interp1d(holo.timestamp, -holo.Theta, fill_value='extrapolate')

head_intp_h = interpolate.interp1d(holo.timestamp, head_h, fill_value='extrapolate')
eye_intp_h = interpolate.interp1d(eye.timestamp, eye_h, fill_value='extrapolate')
Phi_intp = interpolate.interp1d(holo.timestamp, holo.Phi, fill_value='extrapolate')

total_time = np.arange(0, 6.5, 1 / 120)
H_v = head_intp_v(total_time)
E_v = eye_intp_v(total_time)
H_h = head_intp_h(total_time)
E_h = eye_intp_h(total_time)
window = 1.0
watching_period = 0.1 * 120
squares_v = []
Slope_v = []
Intercept_v = []
final_v = []
squares_h = []
Slope_h = []
Intercept_h = []
final_h = []
for t in total_time:
    if t >= 6.5 - window:
        break
    timeline = np.arange(t, t + window, 1 / 120)
    _H_v = head_intp_v(timeline)
    _E_v = eye_intp_v(timeline)
    _H_h = head_intp_h(timeline)
    _E_h = eye_intp_h(timeline)
    rsq_v, itcp_v, slope_v = LinearRegression(_E_v, _H_v)
    rsq_h, itcp_h, slope_h = LinearRegression(_E_h , _H_h)
    squares_v.append(rsq_v)
    Slope_v.append(slope_v)
    Intercept_v.append(itcp_v)
    squares_h.append(rsq_h)
    Slope_h.append(slope_h)
    Intercept_h.append(itcp_h)
    _S_v = sum(Slope_v[-200:]) / len(Slope_v[-200:])
    _I_v = sum(Intercept_v[-200:]) / len(Intercept_v[-200:])
    _S_h = sum(Slope_h[-200:]) / len(Slope_h[-200:])
    _I_h = sum(Intercept_h[-200:]) / len(Intercept_h[-200:])
    # _S_v = sum(Slope_v[int(-watching_period * window):]) / len(Slope_v[int(-watching_period * window):])
    # _I_v = sum(Intercept_v[int(-watching_period * window):]) / len(Intercept_v[int(-watching_period * window):])
    # _S_h = sum(Slope_h[int(-watching_period * window):]) / len(Slope_h[int(-watching_period * window):])
    # _I_h = sum(Intercept_h[int(-watching_period * window):]) / len(Intercept_h[int(-watching_period * window):])
    # main calculation
    # out_v = (head_intp_v(t + window) - (_S_v * (eye_intp_v(t + window))) + _I_v) / 2
    # out_h = (head_intp_h(t + window) - (_S_h * (eye_intp_h(t + window))) + _I_h) / 1
    out_v = (head_intp_v(t + window) - (_S_v * (eye_intp_v(t + window) - _E_v.mean())) + _I_v + _S_v * _E_v.mean()) / 2
    out_h = (head_intp_h(t + window) - (_S_h * (eye_intp_h(t + window) - _E_h.mean())) + _I_h + _S_h * _E_h.mean()) / 2
    final_v.append(out_v)
    final_h.append(out_h)

# fig = make_subplots(rows=2,cols=2,shared_xaxes=True)
# fig.add_trace(
#     go.Scatter(x=total_time,y= H_v,name='Head-Vertical'),row=1,col=1
# )

plt.plot(total_time, H_h)
plt.plot(total_time[int(120 * window):], final_h)
plt.title(str(sum(Slope_h) / len(Slope_h)))
plt.plot(total_time, Phi_intp(total_time))  # Phi
# plt.plot(total_time,H-phi_intp(total_time)) # Target
plt.grid()
plt.show()

# %%
plt.plot(Slope_v)
plt.axhline(sum(Slope_v) / len(Slope_v))
# plt.plot(mean_sl)
plt.show()
plt.plot(Intercept_v)
plt.axhline(sum(Intercept_v) / len(Intercept_v))
plt.show()
plt.scatter(H_v, E_v);
plt.show()
# %%

# new_holo = new_holo[new_holo.timestamp >= 1.5];new_holo.reset_index(inplace=True)
# new_imu = new_imu[new_imu.timestamp >= 1.5];new_imu.reset_index(inplace=True)
# new_eye = new_eye[new_eye.timestamp >= 1.5];new_eye.reset_index(inplace=True)

simulation = overall_algorithm('test', H_offset=120, V_offset=120)

new_holo.head_rotation_x = new_holo.head_rotation_x - new_holo.head_rotation_x[0]
new_holo.head_rotation_y = new_holo.head_rotation_y - new_holo.head_rotation_y[0]
new_eye.norm_x = new_eye.norm_x - new_eye.norm_x[0]
new_eye.norm_y = new_eye.norm_y - new_eye.norm_y[0]
new_eye.theta = new_eye.theta - new_eye.theta[0]

b, a, = signal.iirfilter(2, [0.001, 0.10], btype='bandpass', analog=False, ftype='butter')
targetHorizontal = signal.lfilter(b, a, new_holo.TargetHorizontal)
targetVertical = signal.lfilter(b, a, new_holo.TargetVertical)
# new_eye.norm_y = new_eye.norm_y.rolling(window=10,min_periods=1).mean()
# new_eye.theta = new_eye.theta.rolling(window=10,min_periods=1).mean()
for i in range(new_holo.shape[0]):
    simulation.add_data(new_holo.head_rotation_y[i], new_holo.head_rotation_x[i], new_eye.norm_x[i], new_eye.theta[i])

# %% HORIZONTAL

# H_mults = pd.Series(simulation.H_multiples.copy())
# V_mults = pd.Series(simulation.V_multiples.copy())
# H_mults = H_mults.rolling(window=30,min_periods=1).mean()
# V_mults = V_mults.rolling(window=30,min_periods=1).mean()

plt.plot(new_holo.timestamp[121:], pd.Series(simulation.H_multiples[120:]))
plt.plot(new_holo.timestamp[121:], pd.Series(simulation.V_multiples[120:]))
# plt.plot(new_holo.timestamp[121:], H_mults[120:])
# plt.plot(new_holo.timestamp[121:], V_mults[120:])
plt.show()

mul = [0]
# rol_mul = [0]
for i in range(len(simulation.H_multiples)):
    mul.append(simulation.data[('eye', 'H')].filtered[i + 1] * simulation.H_multiples[i])
    # rol_mul.append(simulation.data[('eye', 'H')].filtered[i + 1] * H_mults[i])
# self.b, self.a = signal.iirfilter(order, [self.lowcut, self.highcut],
#                                   btype='bandpass', analog=False, ftype='butter')

mul = pd.Series(mul)
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y=-targetHorizontal, name='Target'),

        go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'H')].raw, name='raw-head'),
        # go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'H')].filtered, name='filtered-head'),
        go.Scatter(x=new_holo.timestamp, y=mul, name='multiple-eye'),
        go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'H')].raw + mul, name='final'),
        # go.Scatter(x=new_holo.timestamp, y=pd.Series(mul) + new_holo.TargetHorizontal, name='diff')
        go.Scatter(x=new_holo.timestamp, y=new_eye.norm_x, name='raw-eye'),

    ]
)
fig.show()
# %%
import seaborn as sns

offset = 180
temp = pd.DataFrame(data={'raw_head': simulation.data[('head', 'H')].raw[offset:],
                          'raw_eye': simulation.data[('eye', 'H')].raw[offset:],
                          'filter_head': simulation.data[('head', 'H')].filtered[offset:],
                          'filter_eye': simulation.data[('eye', 'H')].filtered[offset:]})
sns.pairplot(temp)
plt.title("H")
plt.show()
# %%
temp = pd.DataFrame(data={'raw_head': simulation.data[('head', 'V')].raw[offset:],
                          'raw_eye': simulation.data[('eye', 'V')].raw[offset:],
                          'filter_head': simulation.data[('head', 'V')].filtered[offset:],
                          'filter_eye': simulation.data[('eye', 'V')].filtered[offset:]})
sns.pairplot(temp)
plt.title("V")
plt.show()
# %% VERTICAL
mul = [0]
# rol_mul=[0]
for i in range(len(simulation.V_multiples)):
    mul.append(simulation.data[('eye', 'V')].filtered[i + 1] * simulation.V_multiples[i])
    # rol_mul.append(simulation.data[('eye', 'V')].filtered[i + 1] * V_mults[i])
mul = pd.Series(mul)
fig = go.Figure(
    data=[
        # go.Scatter(x=new_holo.timestamp, y=-new_holo.TargetVertical.rolling(window=6, min_periods=1).mean(),
        #            name='rolled_Target'),
        go.Scatter(x=new_holo.timestamp, y=targetVertical, name='Target', opacity=0.3),
        # go.Scatter(x=new_holo.timestamp, y=-new_holo.Theta, name='Theta', opacity=0.3),
        # go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'V')].raw, name='raw-head'),
        # go.Scatter(x=new_holo.timestamp, y=pd.Series(simulation.data[('head', 'V')].raw).rolling(window=15,min_periods=1).mean(), name='raw-head-roll'),
        go.Scatter(x=new_holo.timestamp, y=pd.Series(simulation.data[('head', 'V')].filtered), name='filtered-head'),
        go.Scatter(x=new_holo.timestamp, y=mul, name='multiple-eye'),
        go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'V')].raw, name='head'),
        go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'V')].raw - mul, name='final')

    ]
)
fig.show()

# %%
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y=targetVertical, opacity=0.2, name='TargetVertical'),
        go.Scatter(x=new_eye.timestamp, y=-new_eye.norm_y * 250, opacity=0.2, name='eye-raw'),
        go.Scatter(x=new_holo.timestamp, y=new_holo.TargetVertical.rolling(window=10, min_periods=1).mean(),
                   name='roll-TargetVertical'),
        go.Scatter(x=new_eye.timestamp, y=-new_eye.norm_y.rolling(window=10, min_periods=1).mean() * 250,
                   name='roll-eye-raw')
    ]
)
fig.show()
