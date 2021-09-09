# %%
import pandas as pd
from AnalysingFunctions import *
from FileHandling import *
import time

"""
NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi

'timestamp', 'target_entered', 'angular_distance', 'head_position_x',
       'head_position_y', 'head_position_z', 'head_rotation_x',
       'head_rotation_y', 'head_rotation_z', 'head_forward_x',
       'head_forward_y', 'head_forward_z', 'target_position_x',
       'target_position_y', 'target_position_z', 'head_forward_x_next',
       'head_forward_y_next', 'head_forward_z_next', 'time_interval',
       'angle_speed', 'Distance', 'Theta', 'Phi', 'TargetVertical',
       'TargetHorizontal', 'MaximumTargetSize', 'MaximumTargetAngle'
       
       'timestamp', 'target_entered', 'angular_distance', 'head_position_x',
       'head_position_y', 'head_position_z', 'head_rotation_x',
       'head_rotation_y', 'head_rotation_z', 'head_forward_x',
       'head_forward_y', 'head_forward_z', 'target_position_x',
       'target_position_y', 'target_position_z', 'head_forward_x_next',
       'head_forward_y_next', 'head_forward_z_next', 'time_interval',
       'angle_speed', 'Distance', 'Theta', 'Phi', 'TargetVertical',
       'TargetHorizontal', 'MaximumTargetSize', 'MaximumTargetAngle
"""


# %%


#%%
subject = 1
env = 'U'
target = 0
block = 2
posture = 'W'
trial_info = dict(target=target, env=env, posture=posture, block=block,
                  subject=subject, study_num=3)
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture=posture, block=block,
                                                       subject=subject, study_num=3)
eye = eye[eye.confidence > 0.8]
fc = 10

holo = interpolate_dataframe(holo, framerate=120)
eye = interpolate_dataframe(eye, framerate=120)
eye['median_phi'] = eye.phi.rolling(601, min_periods=1).median()
eye['median_theta'] = eye.theta.rolling(601, min_periods=1).median()
eye.phi = eye.phi - eye.median_phi
eye.theta = eye.theta - eye.median_theta
plt.plot(holo.head_position_x,holo.head_position_z)
plt.show()
from filterpy.kalman import KalmanFilter

#%%
# imu = interpolate_dataframe(imu, framerate=60)

import scipy.signal

eye['xhat'] = scipy.signal.savgol_filter(eye.phi,5,2)
eye['yhat'] = scipy.signal.savgol_filter(eye.theta,5,2)
eye['xvel'] = eye.xhat.diff(1)/eye.timestamp.diff(1)
eye['yvel'] = eye.yhat.diff(1)/eye.timestamp.diff(1)
holo['xadd'] = holo.head_rotation_y + eye.xhat
holo['yadd'] = holo.head_rotation_x - eye.yhat
holo['xvel'] = holo.xadd.diff(1) / holo.timestamp.diff(1)
holo['yvel'] = holo.yadd.diff(1) / holo.timestamp.diff(1)
holo['vel'] = (holo.xvel**2+holo.yvel**2).apply(math.sqrt)
plt.plot(holo.xadd)
plt.plot(holo.Phi)
plt.plot(eye.xhat)

plt.show()
plt.plot(holo.yadd)
plt.plot(holo.Theta)
plt.plot(eye.yhat)
plt.show()
eye['vel'] = (eye.xvel**2 + eye.yvel**2).apply(math.sqrt)
plt.plot(eye.vel)
plt.plot(holo.vel,alpha=0.5)
plt.show()
from sklearn.mixture import GaussianMixture
training = eye.vel[1:].to_numpy().reshape(-1,1)
gmm =GaussianMixture(n_components=2).fit(training)

print(gmm.means_)
plt.plot(eye.vel)
plt.plot(gmm.predict(training)*50)
plt.show()

# %% Test
subject = 1
env = 'U'
target = 0
block = 4
posture = 'W'
trial_info = dict(target=target, env=env, posture=posture, block=block,
                  subject=subject, study_num=3)
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture=posture, block=block,
                                                       subject=subject, study_num=3)
# eye = eye[eye.confidence > 0.8]
fc = 10

holo = interpolate_dataframe(holo, framerate=60)
# eye = interpolate_dataframe(eye, framerate=60)
# imu = interpolate_dataframe(imu, framerate=60)
holo['changed_add_angle'] = calculate_anglular_distance(easing_linear(holo.head_rotation_y, 0.08) - holo.Phi,
                                                        easing_linear(holo.head_rotation_x, 0.07) - holo.Theta)
initial_contact_index = holo[holo.timestamp > initial_contact_time].index[0]
changed_initial_contact_time = holo[holo.changed_add_angle < holo.MaximumTargetAngle].timestamp.values[0]
changed_initial_contact_index = holo[holo.timestamp > changed_initial_contact_time].index[0]
plt.plot(holo.timestamp, holo.MaximumTargetAngle)
plt.plot(holo.timestamp, holo.angular_distance, color='red')
plt.plot(holo.timestamp, holo['changed_add_angle'], linestyle='--', color='green')
plt.axvline(initial_contact_time)
plt.show()

# %%
subject = 1
env = 'W'
target = 1
block = 3
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=2)
eye = eye[eye.confidence > 0.8]
fc = 10

holo = interpolate_dataframe(holo, framerate=60)
eye = interpolate_dataframe(eye, framerate=60)
imu = interpolate_dataframe(imu, framerate=60)

eye.phi = eye.phi - eye.phi[0]
eye.theta = eye.theta - eye.theta[0]
eye['median_phi'] = eye.phi.rolling(601, min_periods=1).median()
eye['median_theta'] = eye.theta.rolling(601, min_periods=1).median()
eye.phi = eye.phi - eye.median_phi
eye.theta = eye.theta - eye.median_theta
eye.phi = one_euro(eye.phi, eye.timestamp, 200, 10, 1.0)
eye.theta = one_euro(eye.theta, eye.timestamp, 200, 10, 1.0)

Vmin = 1
Vmax = 20
Dmin = 0.1
Dmax = 1.0
Wmin = 0.005
Wmax = 0.05
Gmin = 0.032
Gmax = 1.055
window = 6


def vhat(v, vmin, vmax):
    if v > vmax:
        return 1
    elif v < vmin:
        return 0
    else:
        return (v - vmin) / (vmax - vmin)


def dhat(d, dmin, dmax):
    if abs(d) > dmax:
        return 1
    elif abs(d) < dmin:
        return 0
    else:
        return (abs(d) - dmin) / (dmax - dmin)


def what(w, wmin, wmax):
    if w > wmax:
        return 1
    elif w < wmin:
        return 0
    else:
        return (w - wmin) / (wmax - wmin)


adaptive_cursor = []
main_data = holo.head_rotation_y
for i in range(len(holo.timestamp)):
    if i <= window:
        adaptive_cursor.append(main_data[i])
    else:
        Vx = (main_data[i] - main_data[i - 1]) / holo.timestamp.diff(1)[i]
        Dx = main_data[i] - adaptive_cursor[i - 1]
        Wx = sum([abs(main_data[i] - main_data[i - x]) for x in range(1, window)]) / window

        Vx_hat = vhat(Vx, Vmin, Vmax)
        Dx_hat = dhat(Dx, Dmin, Dmax)
        Wx_hat = what(Wx, Wmin, Wmax)
        Mx = Wx_hat * (max(Vx_hat, Dx_hat))
        Gx = Gmin + 0.5 * (math.sin(Mx * math.pi - math.pi * 0.5) + 1) * (Gmax - Gmin)
        Sx = main_data[i] - main_data[i - 1]
        if Gx > 1 and Dx > 0 and Sx < 0:
            Gx_hat = 1 - (Gx - 1)
        elif Gx > 1 and Dx < 0 and Sx > 0:
            Gx_hat = 1 - (Gx - 1)
        else:
            Gx_hat = Gx
        # print(Vx_hat,Dx_hat,Wx_hat,Mx,Gx_hat)
        # if holo.timestamp[i] > initial_contact_time:
        #     Gx_hat = 1-Gx_hat
        new_cursor = adaptive_cursor[i - 1] + Gx_hat * Sx
        adaptive_cursor.append(new_cursor)

plt.plot(holo.timestamp, holo.head_rotation_y)

# plt.plot(holo.timestamp, holo.head_rotation_x)
plt.plot(holo.timestamp, adaptive_cursor)
plt.plot(holo.timestamp, holo.Phi, linestyle='--')
plt.axvline(initial_contact_time)
plt.show()

# cursor_x = []
#
# slow = 0.05
# fast = 0.5
# r = 0.5
# for i in range(len(holo.timestamp)):
#     if i == 0:
#         cursor_x.append(holo.head_rotation_y[0])
#     else:
#         if holo.timestamp[i] < initial_contact_time:
#             r = fast
#         else:
#             r = slow
#         previous_cursor = cursor_x[i - 1]
#         eye_cursor = holo.head_rotation_y[i] + eye.phi[i]  # estimation
#         estimated_direction = eye_cursor - holo.head_rotation_y[i]
#         # head_movement = holo.head_rotation_y[i] - holo.head_rotation_y[i - 1]  # head movement
#         head_movement = holo.head_rotation_y.diff(1)[i - 10:i].mean()
#         correct_direction = True if head_movement * estimated_direction > 0 else False
#         if correct_direction:  # head is moving towards
#             if abs(estimated_direction) < abs(eye_cursor - previous_cursor):  # if actual Head is closer than cursor
#                 r = fast
#             else:
#                 r = slow
#         else:
#             r = slow
#         # new_cursor = lerp_one_frame(previous_cursor, holo.head_rotation_y[i],
#         #                             holo.timestamp[i] - holo.timestamp[i - 1], r)
#         new_cursor = previous_cursor * (1-r) + r* holo.head_rotation_y[i]
#         cursor_x.append(new_cursor)


# %%
# whole dataset
subjects = range(1, 17)
envs = ['U', 'W']
targets = range(8)
blocks = range(1, 5)
final_result = []

t = time.time()
for subject, env, target, block in itertools.product(
        subjects, envs, targets, blocks
):
    holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                           subject=subject, study_num=3)
    eye = eye[eye.confidence > 0.8]
    fc = 10
    holo = interpolate_dataframe(holo)
    eye = interpolate_dataframe(eye)
    imu = interpolate_dataframe(imu)
# %% Check stand-condition data
subject = 2
env = 'U'
target = 3
block = 3
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
eye = eye[eye.confidence > 0.8]
fc = 10
holo = interpolate_dataframe(holo)
eye = interpolate_dataframe(eye)
imu = interpolate_dataframe(imu)
holo['added_yaw'] = holo.head_rotation_y + eye.phi
holo['added_pitch'] = holo.head_rotation_x - eye.theta

# speed reduction
# holo = holo[holo.timestamp > 1.0]
# holo['slow_head_rotation_y'] = (holo.head_rotation_y.diff(1)/5).cumsum()
# holo['slow_head_rotation_y'] += holo.head_rotation_y.iloc[0]

# imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = fast
holo['slow_head_rotation_y'] = holo.head_rotation_y
for i, row in holo.iterrows():
    if i == 0:
        continue
    else:
        direction = row['added_yaw'] - row['head_rotation_y']
        cursor_direction = row['added_yaw'] - holo.iloc[i - 1, holo.columns.get_loc('slow_head_rotation_y')]
        current_move = row['head_rotation_y'] - holo.head_rotation_y[i - 1]
        if cursor_direction * current_move < 0:  # opposite direction
            speed = 0.1
        else:
            speed = 1
        holo.iloc[i, holo.columns.get_loc('slow_head_rotation_y')] = holo.iloc[i - 1, holo.columns.get_loc(
            'slow_head_rotation_y')] + speed * current_move
plt.plot(holo.timestamp, holo.head_rotation_y)
plt.plot(holo.timestamp, holo.slow_head_rotation_y)
plt.plot(holo.timestamp, holo.Phi, alpha=0.3)
plt.plot(holo.timestamp, holo.added_yaw, alpha=0.3)
plt.plot(eye.timestamp, eye.phi)
plt.show()
plt.plot(eye.timestamp, eye.phi, color='blue')
from scipy.signal import medfilt

med = medfilt(eye.phi, 601)
plt.plot(eye.timestamp, eye.phi - pd.Series(med), color='green', alpha=0.5)
manual_med = []
for i in range(len(eye)):
    if i < 601:
        strip = eye.phi[:i]
    else:
        strip = eye.phi[i - 201:i]
    manual_med.append(np.median(strip))
plt.plot(eye.timestamp, eye.phi - pd.Series(manual_med), color='red', alpha=0.5)
plt.plot(holo.timestamp, -holo.TargetHorizontal, color='gray')
plt.show()

# %% Check Ground Truth
subject = 16
env = 'U'
target = 3
block = 3
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
eye = eye[eye.confidence > 0.8]
fc = 5
holo = interpolate_dataframe(holo)
eye = interpolate_dataframe(eye)
imu = interpolate_dataframe(imu)
imu['rotationX'] = imu['rotationX'] - imu['rotationX'][0]
imu['rotationZ'] = imu['rotationZ'] - imu['rotationZ'][0]
imu['rotationY'] = imu['rotationY'] - imu['rotationY'][0]
eye['filtered_phi'] = butter_lowpass_filter(eye.phi, fc, 200, real_time=True)
eye['filtered_theta'] = butter_lowpass_filter(eye.theta, fc, 200, real_time=True)
holo['filtered_head_rotation_y'] = butter_lowpass_filter(holo.head_rotation_y, fc, 200, real_time=True)
holo['filtered_head_rotation_x'] = butter_lowpass_filter(holo.head_rotation_x, fc, 200, real_time=True)
# eye.phi = butter_lowpass_filter(eye.phi,fc,200)
# eye.theta = butter_lowpass_filter(eye.theta,fc,200)
from scipy.signal import medfilt

for i in [601]:  # median filter
    Heye_ = eye.phi
    Veye_ = eye.theta
    a = medfilt(Heye_, i)
    b = medfilt(Veye_, i)
    eye['median_phi'] = eye['phi'] - a
    eye['median_theta'] = eye['theta'] - b
    Heye_ = eye.filtered_phi
    Veye = eye.filtered_theta
    a = medfilt(Heye_, i)
    b = medfilt(Veye_, i)
    eye['median_phi_filtered'] = eye['filtered_phi'] - a
    eye['median_theta_filtered'] = eye['filtered_theta'] - b

summary = pd.DataFrame()
summary['timestamp'] = holo.timestamp
summary['H_eye_distance'] = eye['median_phi_filtered'].diff(1)
summary['V_eye_distance'] = eye['median_theta_filtered'].diff(1)
summary['eye_distance'] = (summary['H_eye_distance'] ** 2 + summary['V_eye_distance'] ** 2).apply(math.sqrt)
summary['H_eye_velocity'] = eye['median_phi_filtered'].diff(1).apply(math.sin) / eye.timestamp.diff(1)
summary['V_eye_velocity'] = eye['median_theta_filtered'].diff(1).apply(math.sin) / eye.timestamp.diff(1)
summary['eye_velocity'] = (summary.H_eye_velocity ** 2 + summary.V_eye_velocity ** 2).apply(math.sqrt)

summary['H_head_distance'] = holo['filtered_head_rotation_y'].diff(1)
summary['V_head_distance'] = holo['filtered_head_rotation_x'].diff(1)
summary['head_distance'] = (summary['H_head_distance'] ** 2 + summary['V_head_distance'] ** 2).apply(math.sqrt)
summary['H_head_velocity'] = summary['H_head_distance'].apply(math.sin) / holo.timestamp.diff(1)
summary['V_head_velocity'] = summary['V_head_distance'].apply(math.sin) / holo.timestamp.diff(1)
summary['head_velocity'] = (summary.H_head_velocity ** 2 + summary.V_head_velocity ** 2).apply(math.sqrt)

summary['H_add_distance'] = (eye['median_phi_filtered'] + holo['filtered_head_rotation_y']).diff(1)
summary['V_add_distance'] = (-eye['median_theta_filtered'] + holo['filtered_head_rotation_x']).diff(1)
summary['add_distance'] = (summary['H_add_distance'] ** 2 + summary['V_add_distance'] ** 2).apply(math.sqrt)
summary['H_add_velocity'] = summary['H_add_distance'].apply(math.sin) / holo.timestamp.diff(1)
summary['V_add_velocity'] = summary['V_add_distance'].apply(math.sin) / holo.timestamp.diff(1)
summary['add_velocity'] = (summary.H_add_velocity ** 2 + summary.V_add_velocity ** 2).apply(math.sqrt)
window = 20
window_feature_vectors = []
for i in range(len(summary)):
    if i <= window:
        window_feature_vectors.append(None)
        continue
    eye_angular_distance = summary.eye_distance[i - window:i].mean()
    head_angular_distance = summary.head_distance[i - window:i].mean()
    eye_velocity_deviation = summary.eye_velocity[i - window:i].std()
    head_velocity_deviation = summary.head_velocity[i - window:i].std()
    absolute_eye_velocities = list(summary.eye_velocity[i - window:i])
    absolute_head_velocities = list(summary.head_velocity[i - window:i])
    H_eye_angular_velocity = list(summary.H_eye_velocity[i - window:i])
    V_eye_angular_velocity = list(summary.V_eye_velocity[i - window:i])
    H_head_angular_velocity = list(summary.H_head_velocity[i - window:i])
    V_head_angular_velocity = list(summary.V_head_velocity[i - window:i])
    window_feature_vector = [absolute_eye_velocities, absolute_head_velocities, H_eye_angular_velocity,
                             H_head_angular_velocity, V_eye_angular_velocity, V_head_angular_velocity,
                             eye_angular_distance, head_angular_distance, eye_velocity_deviation,
                             head_velocity_deviation]
    window_feature_vectors.append(window_feature_vector)
summary['window_feature_vector'] = window_feature_vectors
# off = summary.loc[
#     (summary['eye_velocity'] > summary['head_velocity']) & (summary['add_velocity'] > summary['head_velocity'])].index
# plt.plot(summary.timestamp, summary.eye_velocity, label='eye_vel')
# plt.plot(summary.timestamp, summary.head_velocity, label='head_vel')
# for i in summary.timestamp[off].values:
#     plt.axvline(i)
# plt.plot(summary.timestamp, summary.add_velocity.rolling(window, min_periods=1).mean(), label='add_vel')
# plt.axvline(initial_contact_time)
# plt.legend()
# plt.show()
ratio_threshold = 1.5
summary['eye_ratio'] = summary.eye_velocity / summary.head_velocity
summary['add_ratio'] = summary.add_velocity / summary.head_velocity
off = summary.loc[(summary['eye_ratio'] > ratio_threshold) & (summary['add_ratio'] > ratio_threshold)].index
vel_th = 30
sac = summary.loc[(summary['eye_velocity'] > vel_th) & (summary['add_velocity'] > vel_th)].index
head_th = summary.loc[summary['head_velocity'] > 15].index
# plt.plot(summary.timestamp,summary.eye_velocity/summary.head_velocity,label='eye')
# plt.plot(summary.timestamp,summary.add_velocity/summary.head_velocity,label='add')
# plt.ylim(0,3)
from itertools import groupby
from operator import itemgetter

plt.plot(holo.timestamp, summary.head_velocity)
plt.plot(summary.timestamp, summary.add_velocity, alpha=0.5)
plt.scatter(holo.timestamp[off], summary.head_velocity[off], marker='x')
plt.scatter(holo.timestamp[sac], summary.head_velocity[sac], marker='o', color='red')
plt.scatter(holo.timestamp[head_th], summary.head_velocity[head_th], marker='.')
plt.show()
plt.plot(holo.timestamp, holo.filtered_head_rotation_y)
plt.scatter(holo.timestamp[sac], holo.filtered_head_rotation_y[sac], marker='o', color='red')
plt.scatter(holo.timestamp[head_th], holo.filtered_head_rotation_y[head_th], marker='.', color='orange')
plt.plot(holo.timestamp, holo.filtered_head_rotation_x)
plt.scatter(holo.timestamp[sac], holo.filtered_head_rotation_x[sac], marker='o', color='red')
plt.scatter(holo.timestamp[head_th], holo.filtered_head_rotation_x[head_th], marker='.', color='orange')
plt.axvline(initial_contact_time)
plt.show()
# plt.plot(holo.timestamp, holo.head_rotation_x)
# plt.plot(imu.timestamp, imu.rotationX)
# plt.plot(eye.timestamp, eye.median_theta)
# plt.plot(eye.timestamp,holo.head_rotation_x-eye.median_theta,alpha=0.5)
# plt.plot(eye.timestamp,imu.rotationX-eye.median_theta)
# plt.show()
# plt.plot(holo.timestamp, holo.head_rotation_y)
# plt.plot(imu.timestamp, imu.rotationZ)
# plt.plot(eye.timestamp, -eye.median_phi)
# plt.plot(eye.timestamp,holo.head_rotation_y+eye.median_phi,alpha=0.5)
# plt.plot(eye.timestamp,imu.rotationZ+eye.median_phi)
# plt.show()
# angles = get_angle_between_vectors(eye['median_phi'],imu['rotationZ'],-eye['median_theta'],imu['rotationX'])
# angles = get_angle_between_vectors(eye['median_phi_filtered'].diff(1), holo['filtered_head_rotation_y'].diff(1),
#                                    -eye['median_theta_filtered'].diff(1), holo['filtered_head_rotation_x'].diff(1))
# plt.scatter(holo.timestamp[head_th], angles.angle[head_th],marker='.',color='red')
# plt.plot(holo.timestamp, angles.angle)
# plt.show()

# plt.plot(summary['H_eye_velocity']*summary['H_head_velocity'] + summary['V_eye_velocity']*summary['V_head_velocity'])
# plt.show()
print('contact time', initial_contact_time, 'contact frame', len(eye.timestamp[eye.timestamp <= initial_contact_time]))

"""
1 U 3 3 -> 51-55, 125
2 U 3 3 -> 36-40, 186
3 U 3 3 -> 36-40, 104
4 U 3 3 -> 47-53, 159
5 U 3 3 -> 54-58, 171
6 U 3 4 -> 38-42, 286
7 U 3 3 -> 37-41, 161
8 U 3 1 -> 49-53, 162
9 U 3 3 -> 51 55, 90
10 U 3 1 -> 54-59,93
11 U 2 3 -> 32-36, 73
12 U 3 2 -> 37-42, 76
13 U 3 3 -> 34-39, 120
14 U 3 3 -> 53-59, 127
15 U 3 3 -> 18-22, 47
16 U 3 3 -> 58-62, 111
"""
# subject = 3
# env = 'U'
# target = 3
# block = 3
# %%
subject = 5
env = 'U'
target = 3
block = 3
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
eye = eye[eye.confidence > 0.8]
fc = 10
# plt.plot(holo.timestamp, lerp(holo.head_rotation_y, 0.1), label='head_lerp')
# plt.plot(holo.timestamp, lerp(holo.head_rotation_y, holo.timestamp, 1), label='head_lerp')
# plt.plot(holo.timestamp, holo.head_rotation_y, '--', label='head')
# plt.plot(holo.timestamp, holo.Phi, '--', label='target')
original_holo = holo
holo = interpolate_dataframe(holo)
eye = interpolate_dataframe(eye)
imu = interpolate_dataframe(imu)
imu['rotationX'] = imu['rotationX'] - imu['rotationX'][0]
imu['rotationZ'] = imu['rotationZ'] - imu['rotationZ'][0]
imu['rotationY'] = imu['rotationY'] - imu['rotationY'][0]
eye['phi_dejitter'] = double_item_jitter(single_item_jitter(eye.phi))
# fc=20
from scipy.signal import medfilt

for i in [601]:  # median filter
    Heye_ = eye.phi
    Veye_ = eye.theta
    a = medfilt(Heye_, i)
    b = medfilt(Veye_, i)
    eye['median_phi'] = eye['phi'] - a
    eye['median_theta'] = eye['theta'] - b

fcmin = 5
eye.phi = one_euro(eye.median_phi, eye.timestamp, 200, fcmin, 0.1)
eye.theta = one_euro(eye.median_theta, eye.timestamp, 200, fcmin, 0.1)
imu.rotationZ = one_euro(imu.rotationZ, imu.timestamp, 200, fcmin, 0.1)
imu.rotationX = one_euro(imu.rotationX, imu.timestamp, 200, fcmin, 0.1)

# plt.plot(imu.timestamp, imu.rotationZ, label='imu')
# plt.plot(eye.timestamp, eye.phi, label='eye')
# plt.plot(eye.timestamp, imu.rotationZ + eye.phi, label='estimation')
# plt.legend()
# plt.show()
# plt.plot(imu.timestamp, imu.rotationX, label='imu')
# plt.plot(eye.timestamp, -eye.theta, label='eye')
# plt.plot(eye.timestamp, imu.rotationX - eye.theta, label='estimation')
# plt.legend()
# plt.show()

imu_summary = pd.DataFrame()
imu_summary['estimation_H'] = imu.rotationZ + eye.phi
imu_summary['estimation_V'] = imu.rotationX - eye.theta
imu_summary['imu_H'] = imu.rotationZ
imu_summary['imu_V'] = imu.rotationX
imu_summary['timestamp'] = imu.timestamp
slow = 200
fast = 10
imu_summary['lerp_factor'] = fast
imu_summary['H_imu_distance'] = imu.rotationZ.diff(1).apply(math.sin)
imu_summary['H_imu_velocity'] = imu_summary.H_imu_distance / imu.timestamp.diff(1)
imu_summary['V_imu_distance'] = imu.rotationX.diff(1).apply(math.sin)
imu_summary['V_imu_velocity'] = imu_summary.V_imu_distance / imu.timestamp.diff(1)
imu_summary['imu_velocity'] = (imu_summary.H_imu_distance ** 2 + imu_summary.V_imu_distance ** 2).apply(
    math.sqrt) / imu.timestamp.diff(1)
imu_summary['H_eye_distance'] = eye.phi.diff(1).apply(math.sin)
imu_summary['H_eye_velocity'] = imu_summary.H_eye_distance / eye.timestamp.diff(1)
imu_summary['V_eye_distance'] = -eye.theta.diff(1).apply(math.sin)
imu_summary['V_eye_velocity'] = imu_summary.V_eye_distance / eye.timestamp.diff(1)
imu_summary['eye_velocity'] = (imu_summary.H_eye_distance ** 2 + imu_summary.V_eye_distance ** 2).apply(
    math.sqrt) / eye.timestamp.diff(1)

imu_summary['H_add_distance'] = imu_summary.H_eye_distance + imu_summary.H_imu_distance
imu_summary['H_add_velocity'] = imu_summary.H_add_distance / imu.timestamp.diff(1)
imu_summary['V_add_distance'] = imu_summary.V_eye_distance + imu_summary.V_imu_distance
imu_summary['V_add_velocity'] = imu_summary.V_add_distance / imu.timestamp.diff(1)
imu_summary['add_velocity'] = (imu_summary.H_add_distance ** 2 + imu_summary.V_add_distance ** 2).apply(
    math.sqrt) / imu.timestamp.diff(1)
ratio_threshold = 1.5
imu_summary['eye_ratio'] = imu_summary.eye_velocity / imu_summary.imu_velocity
imu_summary['add_ratio'] = imu_summary.add_velocity / imu_summary.imu_velocity

# both eye/add is faster than head
off = imu_summary.loc[(imu_summary['eye_ratio'] < ratio_threshold) & (imu_summary['add_ratio'] < ratio_threshold)].index
vel_th = 30
sac = imu_summary.loc[(imu_summary['eye_velocity'] > vel_th) & (imu_summary['add_velocity'] > vel_th)].index
head_vel_th = 5
head_th = imu_summary.loc[imu_summary['imu_velocity'] > head_vel_th].index

# finding head-following saccade
for i, row in imu_summary.iterrows():
    if len(sac[(i - 50 < sac) & (sac < i)]) > 3:  # if there was saccade before 250 ms
        if row['imu_velocity'] > head_vel_th:
            est_vector = [row['estimation_H'] - row['imu_H'],
                          row['estimation_V'] - row['imu_V']]
            movement_vector = [row['H_imu_velocity'], row['V_imu_velocity']]
            if est_vector[0] * movement_vector[0] + est_vector[1] * movement_vector[1] > 0:
                imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = fast
        else:
            imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = slow
    if row['imu_velocity'] > head_vel_th:
        imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = fast
        est_vector = [row['estimation_H'] - row['imu_H'],
                      row['estimation_V'] - row['imu_V']]
        movement_vector = [row['H_imu_velocity'], row['V_imu_velocity']]
        if est_vector[0] * movement_vector[0] + est_vector[1] * movement_vector[1] > 0:  # go to estimation
            if imu_summary.iloc[i - 1, imu_summary.columns.get_loc('lerp_factor')] == fast:
                imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = fast
        else:
            imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = slow
    else:
        imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = fast

# for i, row in imu_summary.iterrows():
#     if i == 0: continue
#     closer = abs(imu_summary.estimation_H[i - 1] - imu_summary.imu_H[i - 1]) > abs(
#         imu_summary.estimation_H[i] - imu_summary.imu_H[i])
#     if closer:
#         imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = fast
#     else:
#         imu_summary.iloc[i, imu_summary.columns.get_loc('lerp_factor')] = slow
#
# current_cursor = []
# for i in range(len(imu_summary)):
#     if i == 0:
#         current_cursor.append(holo.head_rotation_y[0])
#     else:
#         # dist = current_cursor[i - 1] + (holo.head_rotation_y[i] - holo.head_rotation_y[i - 1]) /3
#         dist = holo.head_rotation_y[i] * imu_summary.lerp_factor[i]
#         current_cursor.append(dist)
imu_summary['current_cursor'] = holo.head_rotation_y
# s = 0.8
dirs = []
head_movement_vectors = []
for i in range(len(imu_summary)):
    if i == 0:
        imu_summary.iloc[i, imu_summary.columns.get_loc('current_cursor')] = holo.head_rotation_y[0]
    else:
        a = holo.head_rotation_y[i]
        previous_cursor = imu_summary.iloc[i - 1]['current_cursor']
        # head_movement_vector = a - previous_cursor
        head_movement_vector = a - holo.head_rotation_y[i - 1]
        eye_movement_vector = eye['phi'][i] - eye['phi'][i - 1]
        dir = eye['phi'][i] + a - previous_cursor
        s = imu_summary.lerp_factor[i]
        # new_cursor = imu_summary.iloc[i - 1]['current_cursor'] + dir * s + head_movement_vector * (1 - s)
        if dir * head_movement_vector >= 0:
            s = 1
        else:
            s = 0.01
        dirs.append(dir)
        head_movement_vectors.append(head_movement_vector)
        new_cursor = previous_cursor + head_movement_vector * s
        imu_summary.iloc[i, imu_summary.columns.get_loc('current_cursor')] = new_cursor

plt.plot(holo.timestamp, holo.head_rotation_y)
plt.plot(holo.timestamp, holo.Phi, '--')
plt.plot(holo.timestamp, holo.head_rotation_y + eye.median_phi)
plt.plot(imu_summary.timestamp, imu_summary.current_cursor)
plt.title(str(s))
plt.show()

plt.plot(head_movement_vectors)
plt.plot(dirs)
plt.plot(pd.Series(head_movement_vectors) * pd.Series(dirs))
plt.show()

plt.plot(eye.timestamp, imu_summary.imu_velocity, label='imu')
# plt.plot(eye.timestamp, imu_summary.eye_velocity, label='eye', alpha=0.5)
plt.axhline(head_vel_th)
# plt.plot(eye.timestamp,imu_summary.add_velocity, label='add',alpha=0.5)
off = imu_summary.loc[(imu_summary['add_ratio'] < ratio_threshold)].index
plt.scatter(eye.timestamp[off], imu_summary.imu_velocity[off], marker='x', color='blue')
plt.scatter(eye.timestamp[sac], imu_summary.imu_velocity[sac], marker='o', color='red')
# plt.scatter(eye.timestamp[head_th], imu_summary.imu_velocity[head_th], marker='.', color='orange')
plt.ylim(0, 50)
plt.legend()
plt.show()

# Ve = eye.phi.diff(1) / eye.timestamp.diff(1)
# Vh = imu.rotationZ.diff(1) / imu.timestamp.diff(1)
# plt.plot(Ve, color='red')
# plt.plot(Vh, color='blue')
# plt.plot(Vh + Ve, color='gray')
# plt.show()

# %% algorithm
imu_summary['saccade'] = False
sac_time_wait = 50  # 50/200 = 250 ms
forward = False
# for i, row in imu_summary.iterrows():
#     # if forward==True:
#     #
#     # else:
#
#     if row['imu_velocity'] < 10:  # normal cursor ( slow head movement )
#         if row['eye_velocity'] > vel_th and row['add_velocity'] > vel_th:
#             # assuming saccade?
#             # imu_summary.set_value(i,'saccade',True)
#             imu_summary.iloc[i, imu_summary.columns.get_loc('saccade')] = True
#         forward = False
#     else:
#         # if row['add_velocity'] < row['imu_velocity']:
#         #     # if added cursor is stable: slow down
#         #     continue
#         if i < sac_time_wait: continue
#         # if there is saccadic eye movement before fast cursor move
#         saccade_prediction = imu_summary['saccade'].iloc[i - sac_time_wait:i]
#         # TODO: verify the movement direction of saccade/head
#         saccade_ratio = saccade_prediction[saccade_prediction == True].count() / sac_time_wait
#         if saccade_ratio > 0.05:
#             forward = True
#             print(i, 'go forward')

# %% Ground Truth calculation
sample_conditions = [(1, 'U', 3, 3, 51, 55),
                     (2, 'U', 3, 3, 36, 40),
                     (3, 'U', 3, 3, 36, 40),
                     (4, 'U', 3, 3, 47, 53),
                     (5, 'U', 3, 3, 54, 58),
                     (6, 'U', 3, 4, 38, 42),
                     (7, 'U', 3, 3, 37, 41),
                     (8, 'U', 3, 1, 49, 53),
                     (9, 'U', 3, 3, 51, 55),
                     (10, 'U', 3, 1, 54, 59),
                     (11, 'U', 2, 3, 32, 36),
                     (12, 'U', 3, 2, 37, 42),
                     (13, 'U', 3, 3, 34, 39),
                     (14, 'U', 3, 3, 53, 59),
                     (15, 'U', 3, 3, 18, 22),
                     (16, 'U', 3, 3, 58, 62)]
vel_fixs = []
vel_saccs = []
vel_purs = []
for condition in sample_conditions:
    holo, eye, imu, initial_contact_time = bring_one_trial(target=condition[2], env=condition[1], posture='W',
                                                           block=condition[3], subject=condition[0])
eye = eye[eye.confidence > 0.8]
ete_index = len(eye.timestamp[eye.timestamp <= initial_contact_time])
eye_dist = (eye.theta.diff(1) ** 2 + eye.phi.diff(1) ** 2).apply(math.sqrt)
eye_vel = (eye.theta.diff(1) ** 2 + eye.phi.diff(1) ** 2).apply(math.sqrt) / eye.timestamp.diff(1)
vel_fix = eye_vel[1:condition[4]]
vel_sacc = eye_vel[condition[4]:condition[5]]
vel_pur = eye_vel[ete_index:]
vel_fixs += list(vel_fix)
vel_saccs += list(vel_sacc)
vel_purs += list(vel_pur)

import plotly.figure_factory as ff

vel_fixs = np.array(vel_fixs)
vel_saccs = np.array(vel_saccs)
vel_purs = np.array(vel_purs)
hist_data = [vel_fixs, vel_saccs, vel_purs]
labels = ['fixation', 'saccade', 'pursuit']
fig = ff.create_distplot(hist_data, labels)
fig.show()

# %%
