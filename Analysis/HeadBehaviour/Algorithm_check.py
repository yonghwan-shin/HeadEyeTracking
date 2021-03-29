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
subject = 1
env = 'U'
target = 3
block = 3
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
eye.theta = double_item_jitter(single_item_jitter(eye.theta))
eye.phi = double_item_jitter(single_item_jitter(eye.phi))
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
subject = 14
env = 'U'
target = 3
block = 2
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
eye = eye[eye.confidence > 0.8]
fc = 5
holo = interpolate_dataframe(holo)
eye = interpolate_dataframe(eye)
imu = interpolate_dataframe(imu)
imu['rotationX'] = imu['rotationX'] - imu['rotationX'][0]
imu['rotationZ'] = imu['rotationZ'] - imu['rotationZ'][0]
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
summary['add_distance'] = (summary['H_add_distance']**2 + summary['V_add_distance']**2).apply(math.sqrt)
summary['H_add_velocity'] = summary['H_add_distance'].apply(math.sin)/holo.timestamp.diff(1)
summary['V_add_velocity'] = summary['V_add_distance'].apply(math.sin)/holo.timestamp.diff(1)
summary['add_velocity'] = (summary.H_add_velocity ** 2 + summary.V_add_velocity**2).apply(math.sqrt)
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

plt.plot(summary.timestamp, summary.eye_velocity.rolling(window,min_periods=1).mean(), label='eye_vel')
plt.plot(summary.timestamp, summary.head_velocity.rolling(window,min_periods=1).mean(), label='head_vel')
plt.plot(summary.timestamp, summary.add_velocity.rolling(window,min_periods=1).mean(), label='add_vel')
plt.axvline(initial_contact_time)
plt.legend()
plt.show()
plt.plot(summary.timestamp,(summary.eye_velocity/summary.head_velocity).rolling(window,min_periods=1).mean(),label='eye/head')
plt.plot(summary.timestamp,summary.add_velocity/summary.head_velocity.rolling(window,min_periods=1).mean(),label='add/head')
plt.legend()
plt.ylim(0,3)
plt.show()
plt.plot(holo.timestamp,holo.head_rotation_x)
plt.plot(imu.timestamp,imu.rotationX)
plt.show()
plt.plot(holo.timestamp,holo.head_rotation_y)
plt.plot(imu.timestamp,imu.rotationZ)
# plt.plot(holo.timestamp,holo.head_rotation_x)
# plt.plot(holo.timestamp,holo.head_rotation_y)
# plt.plot(holo.timestamp,holo.angular_distance)
# plt.axvline(initial_contact_time)
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
Nw = 10

window = 120

# def butter_lowpass_filter(data, cutoff_freq, fs, order=3, real_time=False):

# eye.theta =butter_lowpass_filter(eye.theta,5,120)
# eye.phi =butter_lowpass_filter(eye.phi,5,120)
# eye_dist = (eye.theta.diff(1) ** 2 + eye.phi.diff(1) ** 2).apply(math.sqrt)
# eye_vel = (eye.theta.diff(1) ** 2 + eye.phi.diff(1) ** 2).apply(math.sqrt) / eye.timestamp.diff(1)

# plt.plot(eye.timestamp, -eye.theta)
# plt.plot(eye.timestamp, eye.phi)
# plt.show()
# eye_vel.plot();
# plt.show()
# eye_dist.plot();
# plt.show()

# Vfix = 2
# sig_fix = Vfix * 2 / 3
Vfix = vel_fixs.mean()
sig_fix = Vfix * 2 / 3
# sig_fix = vel_fixs.std()
Vsac = vel_saccs.mean()
# sig_sac = (Vsac * 2) / 3
sig_sac = vel_saccs.std()
print('Vsac', Vsac, 'Ssac', sig_sac, 'Vfix', Vfix)

ris = []

for i in range(len(eye_vel)):
    we = i
    ws = we - Nw
    if ws < 1:
        ws = 1
    ri = eye_vel[i - Nw:i]

    ri = ri[(Vfix < ri) & (ri < Vsac)]  # count between two thresholds
    # ri = ri[(ri > 0) & (ri < Vsac)]
    ris.append(len(ri) / Nw)
plt.plot(ris);
plt.show()

Ppurs = []
Pfixs = []
Psacs = []
result = []


def gauss(value, mean, std):
    import scipy.stats
    return scipy.stats.norm(mean, std).pdf(value)


for i in range(len(eye_vel)):
    # we=i
    # ws=we-Nw
    # if ws<1:
    #     ws=1
    pursuitLike = ris[i]

    Ppur = sum(ris[i - Nw:i - 1]) / (Nw - 1)
    Pfix = (1 - Ppur) / 2
    Psac = Pfix
    if eye_vel[i] < Vfix:
        fixLike = gauss(Vfix, Vfix, sig_fix)
    else:
        fixLike = gauss(eye_vel[i], Vfix, sig_fix)
    if eye_vel[i] < Vsac:
        sacLike = gauss(eye_vel[i], Vsac, sig_sac)
    else:
        sacLike = gauss(Vsac, Vsac, sig_sac)
    evidence = fixLike * Pfix + sacLike * Psac + pursuitLike * Ppur

    fixPosterior = fixLike * Pfix / evidence
    sacPosterior = sacLike * Psac / evidence
    pursuitPosterior = pursuitLike * Ppur / evidence
    idx = np.argmax([fixPosterior, sacPosterior, pursuitPosterior])
    Ppurs.append(pursuitPosterior)
    Pfixs.append(fixPosterior)
    Psacs.append(sacPosterior)
    result.append(idx)
    # if (60 < i < 70):
    #     print(i, eye_vel[i], sacLike, fixLike)
plt.plot(Pfixs[:window], label='fix')
plt.plot(Psacs[:window], label='sac')
plt.plot(Ppurs[:window], label='pur')
plt.plot(eye_vel[:window] / eye_vel[:window].max(), label='vel')
# plt.plot(eye.timestamp[:window], result[:window], label='result')
# plt.plot(eye.timestamp,Ppurs,label='pursuit');
plt.axvline(initial_contact_time)
plt.legend()
plt.show()

# plt.scatter(range(len(eye_vel)), result)
# plt.show()
# result = np.array(result)
# eye_vel.plot()
# plt.axhline(Vsac)
# plt.axhline(Vfix)
# saccades = np.argwhere(result==1)
# for i in saccades:
#     plt.axvline(i)
# plt.show()
