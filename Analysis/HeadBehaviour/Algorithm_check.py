# %%
from AnalysingFunctions import *

from FileHandling import *
import time
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from natsort import natsorted

sns.set_theme(style='whitegrid')
pio.renderers.default = 'browser'
subject = 1
env = 'U'
target = 3
block = 3
cutoff = 0.5
beta = 0.01

output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                            study_num=3)
if env == 'W':
    r = 0.3 / 2
else:
    r = 0.3 / 2
output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
output['one_euro' + str(cutoff) + '_' + str(beta)] = get_new_angular_distance(
    pd.Series(one_euro(output.head_rotation_y, output.timestamp, 60, cutoff, beta)),
    pd.Series(one_euro(output.head_rotation_x, output.timestamp, 60, cutoff, beta)),
    output)
eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                    study_num=3)
print(eye.confidence.mean())
# eye = eye[eye['confidence'] > 0.8]
imu = read_imu_data(target=target, environment=env, posture='W', block=block, subject=subject,
                    study_num=3)
shift, corr, shift_time = synchronise_timestamp(imu, output, show_plot=False)
eye.timestamp = eye.timestamp - shift_time
imu.timestamp = imu.timestamp - shift_time
if env == 'W':
    r = 0.3 / 2
else:
    r = 0.3 / 2
apply = 'angular_distance'
output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
if len(initial_contact_time_data) <= 0:
    print('no contact');
    # continue
initial_contact_time = initial_contact_time_data.timestamp.values[0]
# index = len(Timestamp[Timestamp < initial_contact_time])

eye.theta = (eye.theta - eye.theta[0]) * 360 / (2 * math.pi)
eye.phi = (eye.phi - eye.phi[0]) * 360 / (2 * math.pi)
eye.theta = double_item_jitter(single_item_jitter(eye.theta))
eye.phi = double_item_jitter(single_item_jitter(eye.phi))

# %%
# phi, -theta
# def eye_vel(x,y,t):
Nw = 10


def pow2(x):
    return x ** 2


window = 120

# def butter_lowpass_filter(data, cutoff_freq, fs, order=3, real_time=False):

# eye.theta =butter_lowpass_filter(eye.theta,5,120)
# eye.phi =butter_lowpass_filter(eye.phi,5,120)
eye_dist = (eye.theta.diff(1).apply(pow2) + eye.phi.diff(1).apply(pow2)).apply(math.sqrt)
eye_vel = (eye.theta.diff(1).apply(pow2) + eye.phi.diff(1).apply(pow2)).apply(math.sqrt) / eye.timestamp.diff(1)


plt.plot(eye.timestamp, -eye.theta)
plt.plot(eye.timestamp, eye.phi)
plt.show()
eye_vel.plot();
plt.show()
# eye_dist.plot();
# plt.show()

Vfix = 2
sig_fix = Vfix * 2 / 3
Vsac = 100
sig_sac = Vsac * 2 / 3

ris = []

for i in range(len(eye_vel)):
    ri = eye_vel[i - Nw:i]

    ri = ri[(Vfix < ri) & (ri < Vsac)]
    ris.append(len(ri) / Nw)
# plt.plot(ris);
# plt.show()

Ppurs = []
Pfixs = []
Psacs = []
result = []
for i in range(len(eye_vel)):
    # we=i
    # ws=we-Nw
    # if ws<1:
    #     ws=1
    pursuitLike = ris[i]
    Ppur = sum(ris[i - Nw:i - 1]) / (Nw - 1)
    Pfix = (1 - Ppur) / 2
    Psac = (1 - Ppur) / 2
    if eye_vel[i] < Vfix:
        fixLike = norm_dist(Vfix, Vfix, sig_fix)
    else:
        fixLike = norm_dist(eye_vel[i], Vfix, sig_fix)
    if eye_vel[i] < Vsac:
        sacLike = norm_dist(eye_vel[i], Vsac, sig_sac)
    else:
        sacLike = norm_dist(Vsac, Vsac, sig_sac)
    evidence = fixLike * Pfix + sacLike * Psac + pursuitLike * Ppur
    fixPosterior = fixLike * Pfix / evidence
    sacPosterior = sacLike * Psac / evidence
    pursuitPosterior = pursuitLike * Ppur / evidence
    idx = np.argmax([fixPosterior, sacPosterior, pursuitPosterior])
    Ppurs.append(pursuitPosterior)
    Pfixs.append(fixPosterior)
    Psacs.append(sacPosterior)
    result.append(idx)
plt.plot(eye.timestamp[:window], Pfixs[:window], label='fix')
plt.plot(eye.timestamp[:window], Psacs[:window], label='sac')
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
