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
## Trial parameters
target = 4
env = "W"
block = 2
subject = 303
## Bring the result into pandas dataframe
holo, imu, eye = bring_data(target, env, block, subject)
## Get the delayed time between hololens - laptop
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)

## filter out the low-confidene eye data
# eye = eye[eye.confidence > 0.8]

## match the delayed timestamp into hololens' timestamp
eye.timestamp = eye.timestamp - shift_time
# eye.norm_y = eye.norm_y - eye.norm_y.mean()
## For easy manipulation/check, make all dataframes into 120Hz, and make same timestamp by interpolation (1-dimensional)
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)


#%% success/fail test

subjects = subjects = range(301, 317)
poss = ['W']
envs = ['U','W']
targets=range(8)
# blocks=range(5)
blocks = [1,2,3,4]
output=pd.DataFrame()
success_count=0
fail_count=0
for subject in subjects:
    for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
        holo, imu, eye = bring_data(target, env, block, subject)
        current_info = [target, env, pos, block]
        print(subject,current_info)
        ## Get the delayed time between hololens - laptop
        # shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)

        holo.timestamp = holo.timestamp - holo.timestamp[0]
        # holo = add_columns(holo)
        if holo[holo.target_entered ==True].shape[0]<1:
            print('No target-in');fail_count+=1;continue
        else:
            print('Yes target-in');success_count+=1;continue
        result = dict(
            subject=  subject,
            target=target,
            env=env,
            pos=pos,block=block,
            success_frames=holo[holo.target_entered ==True].shape[0],
            walk_length=holo.head_position_z.values[-1] - holo.head_position_z.values[0]
        )
        output=output.append(result,ignore_index=True)
        # start_index= holo[ holo.target_entered==True].index[0]


# %%
from scipy.signal import lfilter_zi, lfilter


def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, zo = lfilter(b, a, data, zi=zi * data[0])
    return y


def butter_bandpass_filter_zi_realtime(data, highcut, fs, order=2):
    b, a = butter_highpass(highcut, fs, order=order)
    out = []
    z = lfilter_zi(b, a) * data[0]
    for i, D in enumerate(data):
        res, z = lfilter(b, a, data[i:i + 2 * order], zi=z)
        out.append(res[0])
    return pd.Series(out)


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


# eye.norm_y = eye.norm_y - eye.norm_y[0]

# hp_h_rt = butter_bandpass_filter_zi_realtime(holo.head_rotation_y, 0.1, 60, 2)
# highpass_h = [0,0]
# for i in range(2,len(holo.head_rotation_x)):
#     highpass_h.append(murphy_high(0.15,holo.timestamp[i]-holo.timestamp[i-1],
#                                   holo.head_rotation_x[i],holo.head_rotation_x[i-1],holo.head_rotation_x[i-2],
#                                   highpass_h[i-1],highpass_h[i-2]))

def realtime_lowpass(time, sig, cutoff):
    output = [sig[0], sig[1]]
    for i in range(2, len(time)):
        output.append(winter_low(cutoff,
                                 time[i] - time[i - 1],
                                 sig[i], sig[i - 1], sig[i - 2],
                                 output[i - 1], output[i - 2]))
    return output


sig = holo.head_rotation_x
time = holo.timestamp
low_filtered_sig_5 = realtime_lowpass(time, sig, 5)
low_filtered_sig_4 = realtime_lowpass(time, sig, 4)
low_filtered_sig_3 = realtime_lowpass(time, sig, 3)
low_filtered_sig_2 = realtime_lowpass(time, sig, 2)
low_filtered_sig_1 = realtime_lowpass(time, sig, 1)
low_filtered_sig_05 = realtime_lowpass(time, sig, 0.5)
low_filtered_sig_03 = realtime_lowpass(time, sig, 0.3)

# %%
plt.plot(holo.timestamp, holo.head_rotation_x)
fig = go.Figure(
    data=[
        go.Scatter(name='head', x=holo.timestamp, y=holo.head_rotation_x),
        go.Scatter(name='5Hz lowpass', x=holo.timestamp, y=low_filtered_sig_5),
        go.Scatter(name='4Hz lowpass', x=holo.timestamp, y=low_filtered_sig_4),
        go.Scatter(name='3Hz lowpass', x=holo.timestamp, y=low_filtered_sig_3),
        go.Scatter(name='2Hz lowpass', x=holo.timestamp, y=low_filtered_sig_2),
        go.Scatter(name='1Hz lowpass', x=holo.timestamp, y=low_filtered_sig_1),
        go.Scatter(name='0.5Hz lowpass', x=holo.timestamp, y=low_filtered_sig_05),
        go.Scatter(name='0.3Hz lowpass', x=holo.timestamp, y=low_filtered_sig_03),
    ]
)
fig.show()
# %%

one_euro_5 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 5, 0.001, 1.0)
one_euro_4 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 4, 0.001, 1.0)
one_euro_3 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 3, 0.001, 1.0)
one_euro_2 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 2, 0.001, 1.0)
one_euro_1 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 1, 0.001, 1.0)
one_euro_05 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 0.5, 0.001, 1.0)
one_euro_03 = one_euro(holo.head_rotation_x, holo.timestamp, 60, 0.3, 0.001, 1.0)
fig = go.Figure(
    data=[
        go.Scatter(name='head', x=holo.timestamp, y=holo.head_rotation_x),
        go.Scatter(name='one_euro 5Hz', x=holo.timestamp, y=one_euro_5),
        go.Scatter(name='one_euro 4Hz', x=holo.timestamp, y=one_euro_4),
        go.Scatter(name='one_euro 3Hz', x=holo.timestamp, y=one_euro_3),
        go.Scatter(name='one_euro 2Hz', x=holo.timestamp, y=one_euro_2),
        go.Scatter(name='one_euro 1Hz', x=holo.timestamp, y=one_euro_1),
        go.Scatter(name='one_euro 0.5Hz', x=holo.timestamp, y=one_euro_05),
        go.Scatter(name='one_euro 0.3Hz', x=holo.timestamp, y=one_euro_03),
    ]
)
fig.show()
# %%

## Algorithm Test
timeline = np.arange(0, 6.5, 1 / 120)
window = 1.0

# eye_hp = eye.norm_x - one_euro(eye.norm_x, eye.timestamp, 120, 0.02, 0.1, 0.5)
eye_hp = eye.norm_x - one_euro(eye.norm_x, eye.timestamp, 120, 0.2, 0.001, 0.5)
head_hp = holo.head_rotation_y - one_euro(holo.head_rotation_y, holo.timestamp, 60, 0.1, 0.001, 0.5)
e = interpolate.interp1d(eye.timestamp, eye_hp, fill_value='extrapolate')
h = interpolate.interp1d(holo.timestamp, head_hp, fill_value='extrapolate')
H = interpolate.interp1d(holo.timestamp, holo.head_rotation_y, fill_value='extrapolate')
multiple, shift = normalize(e(timeline), h(timeline), RMS=True)
print(multiple)

out = []
muls = []
for t in timeline:
    if t >= 6.5 - window:
        break
    time = np.arange(t, t + window, 1 / 120)
    mul, shi = normalize(e(time), h(time), RMS=True)
    muls.append(mul)
    temp_mul = sum(muls[-int(60):]) / len(muls[-int(60):])
    out.append(H(t + window)
               + temp_mul * e(t + window))

fig = go.Figure(
    data=[
        go.Scatter(x=timeline, y=H(timeline), name='original'),
        go.Scatter(x=timeline, y=h(timeline), name='hp-head'),
        go.Scatter(x=holo.timestamp, y=one_euro(holo.head_rotation_y, holo.timestamp, 60, 0.01, 0.1, 0.5),
                   name='holo-hped'),
        go.Scatter(x=timeline, y=H(timeline) + multiple * e(timeline), name='filtered'),
        go.Scatter(x=timeline, y=multiple * e(timeline), name='RMS'),
        go.Scatter(x=holo.timestamp, y=holo.Phi, name='fit'),
        go.Scatter(x=timeline, y=one_euro(H(timeline) + multiple * e(timeline), timeline, 120, 1, 1.0, 1.0),
                   name='filtered_filtered'),
        go.Scatter(x=timeline[int(120 * window):], y=out, name='final')

    ]
)
fig.show()
# %%


timeline = np.arange(0, 6.5, 1 / 120)
window = 1.0
# eye_hp = eye.norm_y - one_euro(eye.norm_y, eye.timestamp, 120, 0.02, 0.1, 0.5)
eye_hp = eye.norm_y - one_euro(eye.norm_y, eye.timestamp, 120, 0.2, 0.001, 1.0)

head_hp = holo.head_rotation_x - one_euro(holo.head_rotation_x, holo.timestamp, 60, 0.1, 0.001, 1.0)
e = interpolate.interp1d(eye.timestamp, eye_hp, fill_value='extrapolate')
h = interpolate.interp1d(holo.timestamp, head_hp, fill_value='extrapolate')
H = interpolate.interp1d(holo.timestamp, holo.head_rotation_x, fill_value='extrapolate')
multiple, _ = normalize(e(timeline), h(timeline), RMS=True)
print(multiple)
out = []
muls = []
for t in timeline:
    if t >= 6.5 - window:
        break
    time = np.arange(t, t + window, 1 / 120)
    mul, shi = normalize(e(time), h(time), RMS=True)
    muls.append(mul)
    temp_mul = sum(muls) / len(muls)
    out.append(H(t + window)
               + mul * e(t + window))
fig = go.Figure(
    data=[

        go.Scatter(x=timeline, y=H(timeline), name='original'),
        go.Scatter(x=timeline, y=h(timeline), name='hp-head'),
        go.Scatter(x=holo.timestamp, y=one_euro(holo.head_rotation_x, holo.timestamp, 60, 0.1, 0.001, 1.0),
                   name='holo-hped'),
        # go.Scatter(x=timeline, y=H(timeline) + multiple * e(timeline), name='filtered'),
        go.Scatter(x=timeline, y=multiple * e(timeline), name='RMS'),
        go.Scatter(x=timeline, y=one_euro(multiple * e(timeline), timeline, 120, 1, 1.0, 1.0), name='filtered-RMS'),
        # go.Scatter(x=timeline,y=(h(timeline) + multiple*e(timeline))/2,name='average'),
        go.Scatter(x=holo.timestamp, y=-holo.Theta, name='fit'),
        # go.Scatter(x=timeline, y=one_euro(H(timeline) + multiple * e(timeline), timeline, 120, 0.5, 1.0, 1.0),
        #            name='filtered_filtered'),
        go.Scatter(x=timeline[int(120 * window):], y=out, name='final'),
        go.Scatter(x=timeline[int(120 * window):], y=one_euro(out, timeline[int(120 * window):], 120, 3, 0.001, 1.0),
                   name='fff')
    ]
)
fig.show()
# %%
fig = go.Figure(
    data=[
        go.Scatter(x=eye.timestamp, y=eye.norm_y, name='eye-original'),
        go.Scatter(x=eye.timestamp, y=one_euro(eye.norm_y, eye.timestamp, 120, 0.2, 0.001, 1.0), name='eye-0.02'),
        # go.Scatter(x=eye.timestamp,y= one_euro(eye.norm_y, eye.timestamp, 120, 0.1, 0.1, 1.0),name='eye-0.01'),
        go.Scatter(x=eye.timestamp, y=one_euro(eye.norm_y, eye.timestamp, 120, 0.2, 0.01, 1.0), name='eye-0.03'),
        go.Scatter(x=eye.timestamp, y=one_euro(eye.norm_y, eye.timestamp, 120, 0.2, 0.1, 1.0), name='eye-0.03'),
    ]
)
fig.show()

# %% Linear Regression way
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
# watching_period = 120 * 0.05
watching_period = 1 / window
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
    if t >= 6.5 - window:
        break
    timeline = np.arange(t, t + window, 1 / 120)
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
plt.plot(total_time[int(120 * window) + count:], final_h, 'b-')
plt.plot(total_time[int(120 * window) + count:], diff_hs, 'y-')
plt.title(str(watching_period * window) + " H-> " + str(sum(Slope_h) / len(Slope_h)))
plt.plot(total_time, Phi_intp(total_time), 'g-')  # Phi
# plt.plot(total_time,H-phi_intp(total_time)) # Target
plt.grid()
plt.show()

plt.plot(total_time, H_v, 'r-')
plt.plot(total_time[int(120 * window):], final_v, 'b-')
plt.plot(total_time[int(120 * window):], diff_vs, 'y-')
plt.title(str(sum(Slope_v) / len(Slope_v)) + "V")
plt.plot(total_time, theta_intp(total_time), 'g-')  # Phi
# plt.plot(total_time,H-phi_intp(total_time)) # Target
plt.grid()
plt.show()

# %%
plt.plot(Slope_h)
plt.axhline(sum(Slope_h) / len(Slope_h))
# plt.plot(mean_sl)
plt.show()
plt.plot(Intercept_h)
plt.axhline(sum(Intercept_h) / len(Intercept_h))
plt.show()
plt.scatter(H_v, E_v);
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

# %% IIR Filter Test

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
# %% Skim the correlation values
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
# %% SIMULATING --> in no use
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
