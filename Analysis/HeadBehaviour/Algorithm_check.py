# %%
from AnalysingFunctions import *

from FileHandling import *
import time
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from natsort import natsorted

# import remodnav

"""
NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi
"""
sns.set_theme(style='whitegrid')
pio.renderers.default = 'browser'
# subject = 1
# env = 'U'
# target = 3
# block = 3
# holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
#                                                        subject=subject, study_num=3)
# eye.theta = double_item_jitter(single_item_jitter(eye.theta))
# eye.phi = double_item_jitter(single_item_jitter(eye.phi))
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
data = [holo['added_yaw'], holo['added_pitch']]
eye_tsv = pd.concat(data, axis=1)
eye_tsv.to_csv('testADD.tsv', sep='\t', header=None, index=False)

# a = remodnav.main(['remodnav', 'testADD.tsv', 'eventsADD.tsv', '1.0', '200'])

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
th = summary.loc[(summary['eye_velocity'] > vel_th) & (summary['add_velocity'] > vel_th)].index
head_th = summary.loc[summary['head_velocity'] > 15].index
# plt.plot(summary.timestamp,summary.eye_velocity/summary.head_velocity,label='eye')
# plt.plot(summary.timestamp,summary.add_velocity/summary.head_velocity,label='add')
# plt.ylim(0,3)
from itertools import groupby
from operator import itemgetter

plt.plot(holo.timestamp, summary.head_velocity)
plt.plot(summary.timestamp, summary.add_velocity, alpha=0.5)
plt.scatter(holo.timestamp[off], summary.head_velocity[off], marker='x')
plt.scatter(holo.timestamp[th], summary.head_velocity[th], marker='o', color='red')
plt.scatter(holo.timestamp[head_th], summary.head_velocity[head_th], marker='.')
plt.show()
plt.plot(holo.timestamp, holo.filtered_head_rotation_y)
plt.scatter(holo.timestamp[th], holo.filtered_head_rotation_y[th], marker='o', color='red')
plt.scatter(holo.timestamp[head_th], holo.filtered_head_rotation_y[head_th], marker='.', color='orange')
plt.plot(holo.timestamp, holo.filtered_head_rotation_x)
plt.scatter(holo.timestamp[th], holo.filtered_head_rotation_x[th], marker='o', color='red')
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
subject = 2
env = 'U'
target = 3
block = 3
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
eye = eye[eye.confidence > 0.8]
fc = 10
plt.plot(holo.timestamp, lerp(holo.head_rotation_y, 0.03))
plt.plot(holo.timestamp, holo.head_rotation_y, '--')
plt.plot(holo.timestamp, holo.Phi, '--')
holo = interpolate_dataframe(holo)
eye = interpolate_dataframe(eye)
imu = interpolate_dataframe(imu)
imu['rotationX'] = imu['rotationX'] - imu['rotationX'][0]
imu['rotationZ'] = imu['rotationZ'] - imu['rotationZ'][0]
imu['rotationY'] = imu['rotationY'] - imu['rotationY'][0]
# fc=20

fcmin = 5
eye.phi = one_euro(eye.phi, eye.timestamp, 200, fcmin, 0.1)
eye.theta = one_euro(eye.theta, eye.timestamp, 200, fcmin, 0.1)
imu.rotationZ = one_euro(imu.rotationZ, imu.timestamp, 200, fcmin, 0.1)
imu.rotationX = one_euro(imu.rotationX, imu.timestamp, 200, fcmin, 0.1)
# eye.phi = butter_lowpass_filter(eye.phi, fc, 200, 2, True)
# eye.theta = butter_lowpass_filter(eye.theta, fc, 200, 2, True)
# imu.rotationZ = butter_lowpass_filter(imu.rotationZ, fc, 200, 2, True)
# imu.rotationX = butter_lowpass_filter(imu.rotationX, fc, 200, 2, True)
plt.plot(imu.timestamp, imu.rotationZ)
# fact = 0.03 ** (200/60)
# plt.plot(imu.timestamp,lerp(imu.rotationZ,fact))
plt.plot(eye.timestamp, eye.phi)
plt.show()
plt.plot(imu.timestamp, imu.rotationX)
# plt.plot(imu.timestamp,lerp(imu.rotationX,0.03))
plt.plot(eye.timestamp, -eye.theta)
plt.show()
imu_summary = pd.DataFrame()

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
off = imu_summary.loc[(imu_summary['eye_ratio'] > ratio_threshold) & (imu_summary['add_ratio'] > ratio_threshold)].index

vel_th = 30
th = imu_summary.loc[(imu_summary['eye_velocity'] > vel_th) & (imu_summary['add_velocity'] > vel_th)].index
head_th = imu_summary.loc[imu_summary['imu_velocity'] > 10].index
plt.plot(eye.timestamp, imu_summary.imu_velocity, label='imu')
plt.plot(eye.timestamp, imu_summary.eye_velocity, label='eye', alpha=0.5)
# plt.plot(eye.timestamp,imu_summary.add_velocity, label='add',alpha=0.5)
plt.scatter(eye.timestamp[off], imu_summary.imu_velocity[off], marker='x')
plt.scatter(eye.timestamp[th], imu_summary.imu_velocity[th], marker='o', color='red')
plt.scatter(eye.timestamp[head_th], imu_summary.imu_velocity[head_th], marker='.', color='orange')
plt.legend()
plt.show()
imu_summary['saccade'] = False
sac_time_wait = 50
forward = False
for i, row in imu_summary.iterrows():
    if forward==True:
        
    else:

    if row['imu_velocity'] < 10:  # normal cursor ( slow head movement )
        if row['eye_velocity'] > vel_th and row['add_velocity'] > vel_th:
            # assuming saccade?
            # imu_summary.set_value(i,'saccade',True)
            imu_summary.iloc[i, imu_summary.columns.get_loc('saccade')] = True
        forward = False
    else:
        # if row['add_velocity'] < row['imu_velocity']:
        #     # if added cursor is stable: slow down
        #     continue
        if i < sac_time_wait: continue
        # if there is saccadic eye movement before fast cursor move
        saccade_prediction = imu_summary['saccade'].iloc[i - sac_time_wait:i]
        # TODO: verify the movement direction of saccade/head
        saccade_ratio = saccade_prediction[saccade_prediction == True].count() / sac_time_wait
        if saccade_ratio > 0.05:
            forward = True
            print(i, 'go forward')

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
