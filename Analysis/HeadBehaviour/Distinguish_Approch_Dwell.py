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

# %%
subjects = range(1, 17)
envs = ['U', 'W']
targets = range(8)
blocks = range(1, 5)
final_result = []

t = time.time()
#1024 118 8 0.8 confidence
#1024 220 8 0.9 confidence
#1024 350 8 0.95 confidence

for subject, env, target, block in itertools.product(
        subjects, envs, targets, blocks
):

    output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                study_num=3)
    eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    print('mean eye confidence', eye.confidence.mean())
    if eye.confidence.mean() < 0.90:
        print('Too low eye confidence')
        continue
        # return

    # eye = eye[eye['confidence'] > 0.8]

    imu = read_imu_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    shift, corr, shift_time = synchronise_timestamp(imu, output, show_plot=False)
    eye.timestamp = eye.timestamp - shift_time
    imu.timestamp = imu.timestamp - shift_time
    Timestamp = np.arange(0, 6.5, 1 / 120)

    walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
    if (walklength < 3.5):
        print('too short walklength');
        continue
    # Vholo = interpolate.interp1d(output.timestamp, output.head_rotation_x, fill_value='extrapolate')
    # Vimu = interpolate.interp1d(imu.timestamp, imu.rotationX, fill_value='extrapolate')
    # Veye = interpolate.interp1d(eye.timestamp, eye.norm_y, fill_value='extrapolate')
    # Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
    # Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
    # Heye = interpolate.interp1d(eye.timestamp, eye.norm_x, fill_value='extrapolate')
    # AngleSpeed = interpolate.interp1d(output.timestamp, output.angle_speed, fill_value='extrapolate')
    # Hpre_cutoff = 5.0
    # Vpre_cutoff = 5.0

    # continue;

    # if env == 'W':
    #     r = 0.3 / 2
    # else:
    #     r = 0.3 / 2
    # apply = 'angular_distance'
    # output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
    # initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
    # if len(initial_contact_time_data) <= 0:
    #     print('no contact')
    # initial_contact_time = initial_contact_time_data.timestamp.values[0]
    # Vholo = pd.Series(Vholo(Timestamp))
    # Vimu = pd.Series(realtime_lowpass(Timestamp, Vimu(Timestamp), Vpre_cutoff))
    # Veye = pd.Series(realtime_lowpass(Timestamp, Veye(Timestamp), Vpre_cutoff))
    # Hholo = pd.Series(Hholo(Timestamp))
    # Himu = pd.Series(realtime_lowpass(Timestamp, Himu(Timestamp), Hpre_cutoff))
    # Heye = pd.Series(realtime_lowpass(Timestamp, Heye(Timestamp), Hpre_cutoff))
    # vector = (Heye.diff(1) * Himu.diff(1) + Veye.diff(1) * Vimu.diff(1))
    # index = len(Timestamp[Timestamp < initial_contact_time])
    # vector_df = get_angle_between_vectors(Heye.diff(1), Himu.diff(1), Veye.diff(1), Vimu.diff(1))
