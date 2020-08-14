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
# %% Importing
from plotly.subplots import make_subplots
from analysing_functions import (synchronise_timestamp, bring_data, itertools,
                                 go, plt, interpolate, pd, signal, np)
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = 'browser'

# %% test one trial
holo, imu, eye = bring_data(2, "U", 3, 316)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
imu.IMUtimestamp = imu.IMUtimestamp - shift_time
eye.timestamp = eye.timestamp - shift_time

eye.norm_x = eye.norm_x - eye.norm_x.mean()
eye.norm_y = eye.norm_y - eye.norm_y.mean()

interpolation_function_holo_horizontal = interpolate.interp1d(holo.timestamp, holo.head_rotation_y, fill_value='extrapolate')
interpolation_function_imu_horizontal = interpolate.interp1d(imu.IMUtimestamp, imu.rotationZ, fill_value='extrapolate')
interpolation_function_target_horizontal = interpolate.interp1d(holo.timestamp, holo.TargetHorizontal, fill_value='extrapolate')

interpolated_holo_horizontal = pd.Series(interpolation_function_holo_horizontal(eye.timestamp))
interpolated_imu_horizontal = pd.Series(interpolation_function_imu_horizontal(eye.timestamp))
interpolated_target_horizontal = pd.Series(interpolation_function_target_horizontal(eye.timestamp))

interpolation_function_holo_vertical = interpolate.interp1d(holo.timestamp, holo.head_rotation_x, fill_value='extrapolate')
interpolation_function_imu_vertical = interpolate.interp1d(imu.IMUtimestamp, imu.rotationX, fill_value='extrapolate')
interpolation_function_target_vertical = interpolate.interp1d(holo.timestamp, holo.TargetVertical, fill_value='extrapolate')

interpolated_holo_vertical = pd.Series(interpolation_function_holo_vertical(eye.timestamp))
interpolated_imu_vertical = pd.Series(interpolation_function_imu_vertical(eye.timestamp))
interpolated_target_vertical = pd.Series(interpolation_function_target_vertical(eye.timestamp))

fs = 120
fc = 5
w = fc / (fs / 2)
b, a = signal.butter(3, w, 'low', analog=False)
filtered_eye_horizontal = signal.filtfilt(b, a, eye.norm_x)
filtered_eye_vertical = signal.filtfilt(b, a, eye.norm_y)
filtered_target_horizontal = signal.filtfilt(b, a, interpolated_target_horizontal)
filtered_target_vertical = signal.filtfilt(b, a, interpolated_target_vertical)

# get multiple?
# multiple_horizontal = interpolated_target_horizontal.abs().mean() / eye.norm_x.abs().mean()
# multiple_vertical = interpolated_target_vertical.abs().mean() / eye.norm_y.abs().mean()
offset = 180
multiple_horizontal = interpolated_target_horizontal[offset:].abs().mean() / eye.norm_x[offset:].abs().mean()
multiple_vertical = interpolated_target_vertical[offset:].abs().mean() / eye.norm_y[offset:].abs().mean()

# HORIZONTAL
fig_horizontal = make_subplots(rows=4, cols=1)
fig_horizontal.update_layout(title=dict(text=str(multiple_horizontal), font={'size': 30}))
# EYE
fig_horizontal.add_trace(go.Scatter(x=eye.timestamp, y=(eye.norm_x * multiple_horizontal), name='eye-x'), row=1, col=1)
fig_horizontal.add_trace(go.Scatter(x=eye.timestamp, y=(filtered_eye_horizontal * multiple_horizontal), name='filtered-eye-x'), row=1, col=1)
# HOLO
fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_y, name='head-x'), row=2, col=1)
fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.Phi, name='phi-x'), row=2, col=1)
# TARGET
# fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetHorizontal, name='target-x'), row=3, col=1)
fig_horizontal.add_trace(go.Scatter(x=eye.timestamp, y=filtered_target_horizontal, name='filtered-target-x',fill=None), row=3, col=1)
fig_horizontal.add_trace(go.Scatter(x=eye.timestamp,y=-filtered_eye_horizontal*multiple_horizontal,name='filtered-eye-x',fill='tonexty'),row=3,col=1)
# COMPENSATED
fig_horizontal.add_trace(
    go.Scatter(x=eye.timestamp, y=filtered_target_horizontal + filtered_eye_horizontal * multiple_horizontal, name='compensated'), row=4, col=1)
fig_horizontal.show()

# VERTICAL
fig_vertical = make_subplots(rows=4, cols=1)
fig_vertical.update_layout(title=dict(text=str(multiple_vertical), font=dict(size=30)))
# EYE
fig_vertical.add_trace(go.Scatter(x=eye.timestamp, y=(eye.norm_y * multiple_vertical), name='eye-y'), row=1, col=1)
fig_vertical.add_trace(go.Scatter(x=eye.timestamp, y=(filtered_eye_vertical * multiple_vertical), name='filtered-eye-y'), row=1, col=1)
# HOLO
fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_x, name='head-y'), row=2, col=1)
fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.Theta, name='theta-y'), row=2, col=1)
# TARGET
# fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetVertical, name='target-y'), row=3, col=1)
fig_vertical.add_trace(go.Scatter(x=eye.timestamp, y=filtered_target_vertical, name='filtered-target-y',fill=None), row=3, col=1)
fig_vertical.add_trace(go.Scatter(x=eye.timestamp, y=(-filtered_eye_vertical * multiple_vertical), name='eye-y',fill='tonexty'), row=3, col=1)
# COMPENSATED
# fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetVertical, name='diff-y'), row=4, col=1)
fig_vertical.add_trace(go.Scatter(x=eye.timestamp, y=filtered_target_vertical + filtered_eye_vertical * multiple_vertical, name='compensated-y'),
                       row=4,
                       col=1)
fig_vertical.update_layout(
    shapes=[
        dict(type='line',xref='x1',yref='y1',x0=1,y0=0,x1=1,y1=1),
        dict(type='line',xref='x2',yref='y2',x0=1,y0=0,x1=1,y1=1),
        dict(type='line',xref='x3',yref='y3',x0=1,y0=0,x1=1,y1=1),
        dict(type='line',xref='x4',yref='y4',x0=1,y0=0,x1=1,y1=1)
    ]
)
fig_vertical.show()
# %%
# subjects = range(301, 302)
# envs = ["W", "U"]
# targets = range(3)
# blocks = range(2, 3)
# # subjects = range(304, 305)
# # envs = ["W"]
# # targets = range(0, 1)
# # blocks = range(3, 4)
# for subject in subjects:
#     for env, target, block in itertools.product(envs, targets, blocks):
#         try:
#             pass
#         except Exception as e:
#             print(e.args)

# %% fft filtering
sig_fft = fftpack.fft(np.array(interpolated_holo_vertical))
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(interpolated_holo_vertical.size, d=1 / 120)
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('freq (hz)')
plt.ylabel('power')
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]
plt.show()

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)
plt.figure(figsize=(16, 5))
plt.plot(interpolated_holo_vertical, label='original')
plt.plot(filtered_sig, linewidth=3, label='Filtered')
plt.show()

# %%

holo, imu, eye = bring_data(2, "U", 3, 316)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
imu.IMUtimestamp = imu.IMUtimestamp - shift_time
from scipy.spatial.transform import Rotation as R

fig_horizontal = make_subplots(rows=3, cols=1)
fig_horizontal.update_layout(title=dict(text='imu-holo-match', font={'size': 30}))
# EYE
fig_horizontal.add_trace(go.Scatter(x=imu.IMUtimestamp, y=imu.rotationZ, name='imu-x'), row=1, col=1)
fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_y, name='holo-x'), row=1, col=1)

fig_horizontal.add_trace(go.Scatter(x=imu.IMUtimestamp, y=imu.rotationX, name='imu-y'), row=2, col=1)
fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_x, name='holo-y'), row=2, col=1)

fig_horizontal.add_trace(go.Scatter(x=imu.IMUtimestamp, y=imu.rotationY, name='imu-z'), row=3, col=1)
fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_z, name='holo-z'), row=3, col=1)

fig_horizontal.show()
