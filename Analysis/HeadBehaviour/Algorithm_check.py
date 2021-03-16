# %%
from AnalysingFunctions import *

from FileHandling import *
import time
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from natsort import natsorted

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

# %% Check Ground Truth
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

eye_dist = (eye.theta.diff(1) ** 2 + eye.phi.diff(1) ** 2).apply(math.sqrt)
eye_vel = (eye.theta.diff(1) ** 2 + eye.phi.diff(1) ** 2).apply(math.sqrt) / eye.timestamp.diff(1)
eye_vel_filtered = (eye['median_phi_filtered'].diff(1) ** 2 + eye['median_theta_filtered'].diff(
    1) ** 2).apply(math.sqrt) / eye.timestamp.diff(1)
head_dist = (holo.head_rotation_y.diff(1) ** 2 + holo.head_rotation_y.diff(1) ** 2).apply(math.sqrt)
head_vel = (holo.head_rotation_y.diff(1) ** 2 + holo.head_rotation_y.diff(1) ** 2).apply(
    math.sqrt) / holo.timestamp.diff(1)
imu_dist = (imu.rotationX.diff(1) ** 2 + imu.rotationZ.diff(1) ** 2).apply(math.sqrt)
imu_vel = (imu.rotationX.diff(1) ** 2 + imu.rotationZ.diff(1) ** 2).apply(math.sqrt) / imu.timestamp.diff(1)

holo['added_yaw'] = holo.head_rotation_y + eye.median_phi
holo['added_pitch'] = holo.head_rotation_x - eye.median_theta
holo['added_vel'] = (holo['added_yaw'].diff(1) ** 2 + holo['added_pitch'].diff(1) ** 2).apply(
    math.sqrt) / holo.timestamp.diff(1)

holo['added_yaw_filtered'] = butter_lowpass_filter(holo['added_yaw'], fc, 200)
holo['added_pitch_filtered'] = butter_lowpass_filter(holo['added_pitch'], fc, 200)
holo['added_vel_filtered'] = (holo['added_yaw_filtered'].diff(1) ** 2 + holo['added_pitch_filtered'].diff(
    1) ** 2).apply(
    math.sqrt) / holo.timestamp.diff(1)
from scipy.signal import find_peaks

# eye.theta.plot()
# eye.phi.plot()
# plt.title("theta,phi")
# plt.show()
# plt.plot(eye.timestamp, eye_vel)
# plt.axvline(initial_contact_time)
# plt.show()
plt.plot(holo.timestamp, head_vel, label='head_vel', alpha=0.5)
peaks, _ = find_peaks(head_vel, height=20, distance=50)
plt.plot(holo.timestamp[peaks], head_vel[peaks], 'x')
plt.axhline(20)
plt.legend()
plt.show()
# plt.plot(imu.timestamp, imu_vel)
# plt.plot(eye.timestamp, eye_vel, label='eye_vel', alpha=0.5)
plt.plot(eye.timestamp, eye_vel_filtered, label='eye_vel_filtered')
peaks, _ = find_peaks(eye_vel_filtered, height=20, distance=20)
plt.plot(eye.timestamp[peaks], eye_vel_filtered[peaks], 'x')
plt.show()
# plt.plot(holo.timestamp, holo.added_vel, label='added_vel', alpha=0.5)
plt.plot(holo.timestamp, holo.added_vel_filtered, label='added_vel_filtered')
peaks,_=find_peaks(holo.added_vel_filtered,height = 20,distance = 50)
plt.plot(holo.timestamp[peaks],holo.added_vel_filtered[peaks],'x')
plt.axvline(initial_contact_time)
plt.legend()
plt.show()
plt.plot(holo.timestamp, holo.head_rotation_y)
plt.plot(holo.timestamp, butter_lowpass_filter(holo.head_rotation_y, fc, 200), linestyle='--')
plt.plot(eye.timestamp, eye.phi, alpha=0.2)
plt.plot(eye.timestamp, eye.median_phi)
plt.plot(holo.timestamp, holo.added_yaw, alpha=0.4)
plt.plot(holo.timestamp, butter_lowpass_filter(holo.added_yaw, fc, 200))

plt.plot(holo.timestamp, -holo.TargetHorizontal, alpha=0.3)
plt.plot(holo.timestamp, holo.Phi, alpha=0.3)
plt.axvline(initial_contact_time)
plt.show()
plt.plot(holo.timestamp, holo.head_rotation_x)
plt.plot(holo.timestamp, butter_lowpass_filter(holo.head_rotation_x, fc, 200), linestyle='--')
plt.plot(eye.timestamp, -eye.theta, alpha=0.2)
plt.plot(eye.timestamp, -eye.median_theta)
plt.plot(holo.timestamp, holo.added_pitch, alpha=0.4)
plt.plot(holo.timestamp, butter_lowpass_filter(holo.added_pitch, fc, 200))
plt.plot(holo.timestamp, -holo.TargetVertical, alpha=0.3)
plt.plot(holo.timestamp, holo.Theta, alpha=0.3)
plt.axvline(initial_contact_time)
plt.show()

# lookup = 120
# xs = list(range(lookup))
# fig = go.Figure(
#     data=[
#         go.Scatter(x=xs, y=eye.theta[:lookup], name='theta'),
#         go.Scatter(x=xs, y=eye.phi[:lookup], name='phi'),
#     ]
# )
# fig.show()
# fig = go.Figure(
#     data=[
#         go.Scatter(x=xs, y=eye_vel[:lookup])
#     ]
# )
# fig.show()
# fig = go.Figure(
#     data=[
#         go.Scatter(x=holo.timestamp,y=holo.head_rotation_y,name='yaw'),
#         go.Scatter(x=holo.timestamp,y=holo.head_rotation_x,name='pitch'),
#     ]
# )
# fig.show()
print(initial_contact_time, len(eye.timestamp[eye.timestamp <= initial_contact_time]))

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
Nw = 5

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
