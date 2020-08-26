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
from analysing_functions import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = 'browser'

# %% test one trial
holo, imu, eye = bring_data(0, "U", 4, 316)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
imu.IMUtimestamp = imu.IMUtimestamp - shift_time
eye.timestamp = eye.timestamp - shift_time
eye.norm_x = eye.norm_x - eye.norm_x.mean()
eye.norm_y = eye.norm_y - eye.norm_y.mean()

# imu = imu_to_vector(imu)
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
new_imu = imu_to_vector(new_imu)
new_holo = holo_to_vector(new_holo)
# new_holo = new_holo[new_holo.timestamp >= 1.5]
# new_imu = new_imu[new_imu.timestamp >= 1.5]
# new_eye = new_eye[new_eye.timestamp >= 1.5]
rot, rmsd = R.align_vectors(new_holo[['head_forward_x', 'head_forward_y', 'head_forward_z']], new_imu[['vector_x', 'vector_y', 'vector_z']])
applied_imu = rot.apply(new_imu[['vector_x', 'vector_y', 'vector_z']])

fs = 120
fc = 4
w = fc / (fs / 2)
mincutoff = 3.0
beta = 0.01
b, a = signal.butter(3, w, 'low', analog=False)
# filtered_norm_x= pd.Series(signal.filtfilt(b, a, new_eye.norm_x))
filtered_norm_x = pd.Series(one_euro(new_eye.norm_x, beta=beta, mincutoff=mincutoff))
filtered_norm_x.index = new_eye.index
new_eye['filtered_norm_x'] = filtered_norm_x
filtered_norm_y = pd.Series(one_euro(new_eye.norm_y, beta=beta, mincutoff=mincutoff))
filtered_norm_y.index = new_eye.index
new_eye['filtered_norm_y'] = filtered_norm_y
filtered_target_horizontal = pd.Series(one_euro(new_holo.TargetHorizontal, beta=beta, mincutoff=mincutoff))
filtered_target_horizontal.index = new_holo.index
new_holo['filtered_TargetHorizontal'] = filtered_target_horizontal
filtered_target_vertical = pd.Series(one_euro(new_holo.TargetVertical, beta=beta, mincutoff=mincutoff))
filtered_target_vertical.index = new_holo.index
new_holo['filtered_TargetVertical'] = filtered_target_vertical

# get multiple?
# multiple_horizontal = interpolated_target_horizontal.abs().mean() / eye.norm_x.abs().mean()
# multiple_vertical = interpolated_target_vertical.abs().mean() / eye.norm_y.abs().mean()
offset = int(120 * 1.5)
multiple_horizontal = new_holo.filtered_TargetHorizontal[offset:].abs().mean() / new_eye.filtered_norm_x[offset:].abs().mean()
multiple_vertical = new_holo.filtered_TargetVertical[offset:].abs().mean() / new_eye.filtered_norm_y[offset:].abs().mean()

# HORIZONTAL
fig_horizontal = make_subplots(rows=4, cols=1)
fig_horizontal.update_layout(title=dict(text=str(multiple_horizontal), font={'size': 30}))
# EYE
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=(new_eye.norm_x * multiple_horizontal), name='eye-x', opacity=0.5), row=1, col=1)
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=(new_eye.filtered_norm_x * multiple_horizontal), name='filtered-eye-x'), row=1, col=1)
# HOLO
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetHorizontal, name='filtered_target-x'), row=2, col=1)
# slope_horizontal, intercept_horizontal = normalize(new_imu.rotationZ, new_holo.head_rotation_y)
# fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_imu.rotationZ * slope_horizontal + intercept_horizontal, name='imu-x'), row=2, col=1)
fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_y - holo.Phi, name='target-x'), row=2, col=1)
# TARGET
# fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetHorizontal, name='target-x'), row=3, col=1)
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetHorizontal, name='filtered-target-x'), row=3, col=1)
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.TargetHorizontal, name='target-x', opacity=0.3), row=3, col=1)
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=-new_eye.filtered_norm_x * multiple_horizontal, name='filtered-eye-x'),
                         row=3,
                         col=1)
# COMPENSATED
fig_horizontal.add_trace(
    go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetHorizontal + new_eye.filtered_norm_x * multiple_horizontal, name='compensated'), row=4,
    col=1)

# VERTICAL
fig_vertical = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=("eye", 'head', 'target', 'compensation'))
fig_vertical.update_layout(title=dict(text=str(multiple_vertical), font=dict(size=30)))
# EYE
fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=(new_eye.norm_y * multiple_vertical), name='eye-y', opacity=0.5), row=1, col=1)
fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=(new_eye.filtered_norm_y * multiple_vertical), name='filtered-eye-y'), row=1, col=1)
# HOLO
# fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.head_rotation_x, name='head-y'), row=2, col=1)
fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetVertical, name='filtered-target-y'), row=2, col=1)
# fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_imu.rotationX, name='imu-y'), row=2, col=1)
fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.Theta + holo.head_rotation_x, name='target-y', opacity=0.3), row=2, col=1)
# TARGET
# fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetVertical, name='target-y'), row=3, col=1)
fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetVertical, name='filtered-target-y'), row=3, col=1)
fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=(-new_eye.filtered_norm_y * multiple_vertical), name='eye-y'), row=3, col=1)
# fill='tonexty
# COMPENSATED
fig_vertical.add_trace(
    go.Scatter(x=new_eye.timestamp, y=new_holo.TargetVertical + new_eye.norm_y * multiple_vertical, name='unfiltered-compensate', opacity=0.3),
    row=4, col=1)
fig_vertical.add_trace(
    go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetVertical + new_eye.filtered_norm_y * multiple_vertical, name='compensated-y'),
    row=4,
    col=1)
fig_vertical.update_yaxes(range=[-2, 2], row=1, col=1)
fig_vertical.update_yaxes(range=[-2, 2], row=2, col=1)
fig_vertical.update_yaxes(range=[-2, 2], row=3, col=1)
fig_vertical.update_yaxes(range=[-2, 2], row=4, col=1)
window = 120
estimation_horizontal = []
estimation_vertical = []
for index in range(new_eye.shape[0] - window - 1):
    temp_holo = new_holo[index:index + window]
    temp_eye = new_eye[index:index + window]
    slope_horizontal, intercept_horizontal = normalize(temp_eye.filtered_norm_x, temp_holo.filtered_TargetHorizontal)
    slope_vertical, intercept_vertical = normalize(temp_eye.filtered_norm_y, temp_holo.filtered_TargetVertical)
    estimated_horizontal = new_eye.iloc[index + window + 1]['filtered_norm_x'] * slope_horizontal + intercept_horizontal
    estimated_vertical = new_eye.iloc[index + window + 1]['filtered_norm_y'] * slope_vertical + intercept_vertical
    estimation_horizontal.append(estimated_horizontal)
    estimation_vertical.append(estimated_vertical)
fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp[window + 1:], y=estimation_horizontal, name='real-time-x'), row=3, col=1)
fig_horizontal.add_trace(
    go.Scatter(x=new_eye.timestamp[window + 1:], y=new_holo.filtered_TargetHorizontal[window + 1:] - estimation_horizontal, name='estimation-x'),
    row=4, col=1)
fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp[window + 1:], y=estimation_vertical, name='real-time-y'), row=3, col=1)
fig_vertical.add_trace(
    go.Scatter(x=new_eye.timestamp[window + 1:], y=new_holo.filtered_TargetVertical[window + 1:] - estimation_vertical, name='estimation-y'),
    row=4, col=1)
fig_horizontal.show()
fig_vertical.show()
# %%
plt.plot(new_holo.filtered_TargetVertical, new_eye.filtered_norm_y * multiple_vertical)
plt.plot(new_holo.filtered_TargetVertical[window + 1:], estimation_vertical)
plt.show()
# %%
import plotly.figure_factory as ff

hist_data = [new_holo.filtered_TargetHorizontal + new_eye.filtered_norm_x * multiple_horizontal,
             new_holo.filtered_TargetVertical + new_eye.filtered_norm_y * multiple_vertical,
             new_holo.TargetHorizontal, new_holo.TargetVertical]
group_labels = ['horizontal', 'vertical', 'target_horizontal', 'target_vertical']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2, show_hist=False, curve_type='normal')
fig.show()

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
# %%

from plotly.subplots import make_subplots
from analysing_functions import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = 'browser'
subjects = range(301, 317)
envs = ["W", "U"]
targets = range(8)
blocks = range(1, 5)

W_startpositionX = []
W_startAngles = []
W_walklengths=[]
U_startpositionX = []
U_startAngles = []
U_walklengths=[]
for subject in subjects:
    for env, target, block in itertools.product(['W'], targets, blocks):
        try:
            holo = bring_hololens_data(target, env, block, subject)

            W_startpositionX.append(holo.iloc[5]['head_position_x'])
            W_startAngles.append(holo.iloc[5]['angular_distance'])
            walk = holo['head_position_z'].iloc[-1] - holo['head_position_z'].iloc[0]
            W_walklengths.append(walk)
            print(env, target, block, subject)
            pass
        except Exception as e:
            print(e.args)
for subject in subjects:
    for env, target, block in itertools.product(['U'], targets, blocks):
        try:
            holo = bring_hololens_data(target, env, block, subject)

            U_startpositionX.append(holo.iloc[5]['head_position_x'])
            U_startAngles.append(holo.iloc[5]['angular_distance'])
            walk = holo['head_position_z'].iloc[-1] - holo['head_position_z'].iloc[0]
            U_walklengths.append(walk)
            print(env, target, block, subject)
            pass
        except Exception as e:
            print(e.args)
#%%
from scipy.stats import ks_2samp,wasserstein_distance
print(ks_2samp(W_startpositionX,W_startAngles))
print(ks_2samp(W_startpositionX,U_startAngles))
print(ks_2samp(W_startpositionX,U_startpositionX))
print(ks_2samp(W_startpositionX,W_startpositionX))
print(ks_2samp(U_startpositionX,W_startAngles))
print(ks_2samp(U_startpositionX,U_startAngles))
print(ks_2samp(U_startpositionX,U_startpositionX))
print(ks_2samp(U_startpositionX,W_startpositionX))

#%%
import plotly.figure_factory as ff

fig = ff.create_distplot([W_startpositionX,U_startpositionX],['W','U'],bin_size=0.02,show_rug=False)
fig.show()
W_startAngles = pd.Series(W_startAngles)
W_startAngles = W_startAngles[W_startAngles<150]
fig = ff.create_distplot([W_startAngles,U_startAngles],['W','U'],bin_size=0.2,show_rug=False)
fig.show()
