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
#%%
vector_df = pd.read_pickle('vector_data.pkl')
# %%
subjects = range(1, 17)
envs = ['U']
targets = range(8)
blocks = range(1, 5)
final_result = []

t = time.time()
# 1024 118 8 0.8 confidence
# 1024 220 8 0.9 confidence
# 1024 350 8 0.95 confidence

# for lag, threshold in itertools.product([20, 30, 40, 50], [5, 6, 7, 8, 9, 10]):
    # approachs = []
    # dwells = []
df = []
for subject, env, target, block in itertools.product(
        subjects, envs, targets, blocks
):

    output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                study_num=3)
    eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    # print('mean eye confidence', eye.confidence.mean())
    if eye.confidence.mean() < 0.90:
        # print('Too low eye confidence')
        continue
        # return

    eye = eye[eye['confidence'] > 0.8]

    imu = read_imu_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    # shift, corr, shift_time = synchronise_timestamp(imu, output, show_plot=False)
    # eye.timestamp = eye.timestamp - shift_time
    # imu.timestamp = imu.timestamp - shift_time
    Timestamp = np.arange(0, 6.5, 1 / 200)

    walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
    if (walklength < 3.5):
        # print('too short walklength');
        continue

    HTarget = interpolate.interp1d(output.timestamp, output.Phi, fill_value='extrapolate')
    VTarget = interpolate.interp1d(output.timestamp, output.Theta, fill_value='extrapolate')

    Vholo = interpolate.interp1d(output.timestamp, output.head_rotation_x, fill_value='extrapolate')
    Vimu = interpolate.interp1d(imu.timestamp, imu.rotationX, fill_value='extrapolate')
    # Veye = interpolate.interp1d(eye.timestamp, eye.norm_y, fill_value='extrapolate')
    Veye = interpolate.interp1d(eye.timestamp, -eye.theta, fill_value='extrapolate')
    Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
    Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
    # Heye = interpolate.interp1d(eye.timestamp, eye.norm_x, fill_value='extrapolate')
    Heye = interpolate.interp1d(eye.timestamp, eye.phi, fill_value='extrapolate')
    AngleSpeed = interpolate.interp1d(output.timestamp, output.angle_speed, fill_value='extrapolate')
    Hpre_cutoff = 5.0
    Vpre_cutoff = 5.0

    if env == 'W':
        r = 0.3 / 2
    else:
        r = 0.3 / 2
    apply = 'angular_distance'
    output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
    initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
    if len(initial_contact_time_data) <= 0:
        # print('no contact');
        continue
    initial_contact_time = initial_contact_time_data.timestamp.values[0]
    VTarget = pd.Series(realtime_lowpass(Timestamp, VTarget(Timestamp), Vpre_cutoff))
    Vholo = pd.Series(realtime_lowpass(Timestamp, Vholo(Timestamp), Vpre_cutoff))
    Vimu = pd.Series(realtime_lowpass(Timestamp, Vimu(Timestamp), Vpre_cutoff))
    Veye = pd.Series(realtime_lowpass(Timestamp, Veye(Timestamp), Vpre_cutoff))
    HTarget = pd.Series(realtime_lowpass(Timestamp, HTarget(Timestamp), Hpre_cutoff))
    Hholo = pd.Series(realtime_lowpass(Timestamp, Hholo(Timestamp), Hpre_cutoff))
    Himu = pd.Series(realtime_lowpass(Timestamp, Himu(Timestamp), Hpre_cutoff))
    Heye = pd.Series(realtime_lowpass(Timestamp, Heye(Timestamp), Hpre_cutoff))
    vector = (Heye.diff(1) * Himu.diff(1) + Veye.diff(1) * Vimu.diff(1))
    vector = vector * 200 * 200
    index = len(Timestamp[Timestamp < initial_contact_time])
    df.append(dict(
        subject=subject, env=env, target=target, block=block,
        vector = vector,index=index
    ))
vector_df  = pd.DataFrame(df)
vector_df.to_pickle("vector_data.pkl")
    # lag = 20
    # threshold = 10
    # peak_detection = real_time_peak_detection(array=vector[:lag], lag=lag, threshold=threshold, influence=0.10)
    # output = [0] * lag
    # for n, i in enumerate(vector[lag:]):
    #     # if i < 0: i = 0
    #
    #     p, avg, dev = peak_detection.thresholding_algo(i)
    #     output.append(p)
    # approachs.append(output[:index].count(1))
    # dwells.append(output[index:].count(1))
# approachs = [item for sublist in approachs for item in sublist]
# dwells = [item for sublist in dwells for item in sublist]
# sns.histplot(approachs, bins=50);
# sns.histplot(dwells, bins=50, color='red')
# plt.title("lag: " + str(lag) + " threshold: " + str(threshold))
# plt.show()
# TN = 0
# FN = 0
# FP = 0
# TP = 0
# for i in range(len(dwells)):
#     if approachs[i] > 0:
#         TP += 1
#     if dwells[i] <= 0:
#         TN += 1
#     if approachs[i] <= 0:
#         FP += 1
#     if dwells[i] > 0:
#         FN += 1
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# F1 = 2 * (precision * recall) / (precision + recall)
# print('lag', lag, 'threshold', threshold)
# print('TN, FN, FP, TP', TN, FN, FP, TP)
# print('precision', precision, 'recall', recall, "F1", F1)
