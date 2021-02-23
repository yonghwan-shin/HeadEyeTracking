# %%
from AnalysingFunctions import *

from FileHandling import *
import time
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from natsort import natsorted
import scipy

sns.set_theme(style='whitegrid')
pio.renderers.default = 'browser'

# %%
subjects = [1]
envs = ['U']
targets = [6]
blocks = [3]
# subjects = range(1, 17)
# envs = ['U']
# targets = range(8)
# blocks = range(1, 5)
final_result = []
final_rs = []
final_ps = []
t = time.time()
final_df = []
for subject, env, target, block in itertools.product(
        subjects, envs, targets, blocks
):
    output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                study_num=3)
    eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    print(subject,env,target,block ,'mean eye confidence', eye.confidence.mean())
    if eye.confidence.mean() < 0.95:
        print('Too low eye confidence')
        continue
    walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
    if (walklength < 3.5):
        print('too short walklength');
        continue;
    eye = eye[eye['confidence'] > 0.8]
    imu = read_imu_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    shift, corr, shift_time = synchronise_timestamp(imu, output, show_plot=False)
    eye.timestamp = eye.timestamp - shift_time
    imu.timestamp = imu.timestamp - shift_time
    Timestamp = np.arange(0, 6.5, 1 / 200)
    HTarget = interpolate.interp1d(output.timestamp, output.Phi, fill_value='extrapolate')
    VTarget = interpolate.interp1d(output.timestamp, output.Theta, fill_value='extrapolate')
    Vholo = interpolate.interp1d(output.timestamp, output.head_rotation_x, fill_value='extrapolate')
    Vimu = interpolate.interp1d(imu.timestamp, imu.rotationX, fill_value='extrapolate')
    Veye = interpolate.interp1d(eye.timestamp, -eye.theta, fill_value='extrapolate')
    Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
    Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
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
        print('no contact');
        continue
    initial_contact_time = initial_contact_time_data.timestamp.values[0]
    index = len(Timestamp[Timestamp < initial_contact_time])
    HTarget = pd.Series(HTarget(Timestamp))
    VTarget = pd.Series(VTarget(Timestamp))
    Vholo = pd.Series(Vholo(Timestamp))
    Veye = pd.Series(Veye(Timestamp))
    Hholo = pd.Series(Hholo(Timestamp))
    Heye = pd.Series(Heye(Timestamp))
    Heye = Heye * 180 / math.pi
    Veye = Veye * 180 / math.pi



# %%
data = pd.read_pickle('interpolated_summary.pkl')
for index, row in data.iterrows():
    if index == 52:
        fc = 5
        w = fc / (200 / 2)
        b, a = scipy.signal.butter(2, w, 'low')
        fch = 0.20
        wh = fch / (200 / 2)
        hb, ha = scipy.signal.butter(2, wh, 'highpass')
        highfc=0.1
        highw = highfc/(200/2)
        highb,higha = scipy.signal.butter(2,highw,'high')
        row['Heye'] = pd.Series(signal.filtfilt(b, a, row['Heye']))
        zi = signal.lfilter_zi(highb, higha) * row['Heye'][0]

        lfilter_Heye = []

        for x in list(row['Heye']):
            y, zi = signal.lfilter(highb, higha, [x], zi=zi)
            lfilter_Heye.append(y[0])

        t = row['Hholo'] - row['HTarget']
        lfilter_Heye = pd.Series(lfilter_Heye)
        filtfilt_Heye = signal.filtfilt(hb,ha,row['Heye'])
        mean_Heye=  row['Heye']-row['Heye'][0]
        # roll_Heye=row['Heye']-row['Heye'].rolling(400,min_periods=1).mean()
        roll_Heye=signal.detrend(row['Heye'])
        plt.plot(-t,color='blue',label='target')
        # plt.plot(lfilter_Heye,color='orange',label='lfilter')
        # plt.plot(filtfilt_Heye,color='green',label='filtfilt')
        plt.plot(mean_Heye,color='red',label='mean')
        plt.plot(roll_Heye,color='pink',label='roll')
        plt.plot(t+mean_Heye,color='cyan')
        plt.plot(t + roll_Heye, color='gray')
        # plt.plot(signal.filtfilt(b,a,t+mean_Heye),color='cyan')
        plt.legend()
        plt.axvline(row['first_index'])
        plt.ylim(-3,3)
        plt.show()
        # row['Himu'] =pd.Series( row['Himu'](np.arange(0,6.5,1/200)))
        # plt.scatter(row["Himu"].diff(1),pd.Series(row["Heye"]).diff(1))
        # # plt.plot(pd.Series(row["Heye"]).diff(1), label='eye vel')
        # plt.show()
        import plotly.figure_factory as ff

        hist_data = [t[row['first_index']:], (t + lfilter_Heye)[row['first_index']:],
                     (t + filtfilt_Heye)[row['first_index']:], (t + row['Heye']-row['Heye'].mean())[row['first_index']:]]
        labels = ['target', 'lfilter_Heye', 'filtfilt_Heye', 'comp']
        fig = ff.create_distplot(hist_data, labels, bin_size=.01)
        fig.update_layout(title="t:" + str(hist_data[0].std()) + " ,Eye: " + str(hist_data[1].std()) + " ,comp: " + str(
            hist_data[2].std()) + " hp cut:" + str(fch))
        fig.show()
        print(row['first_index'],row['target'])
# %%

# %%
a = []
b = []
for i in range(len(final_result)):
    a = a + list(final_result[i][0])
    b = b + list(final_result[i][1])
rr_t = []
pp_t = []
rr = []
pp = []

for i in range(len(final_rs)):
    rr_t = rr_t + final_rs[i][0]
    rr = rr + final_rs[i][1]
    pp_t = pp_t + final_ps[i][0]
    pp = pp + final_ps[i][1]
# pd.Series(a)
# pd.Series(b)

sns.distplot(pd.Series(a), bins=100, color='red', label='H')
sns.distplot(pd.Series(b), bins=400, color='blue', label='H+E')
plt.legend()
plt.xlim(-5, 5)
plt.show()

sns.distplot(pd.Series(pp), bins=100, color='red', label='pvalue_dwell')
sns.distplot(pd.Series(pp_t), bins=100, color='blue', label='pvalue_target')
plt.legend()
plt.show()
sns.distplot(pd.Series(rr), bins=100, color='red', label='rvalue_dwell')
sns.distplot(pd.Series(rr_t), bins=100, color='blue', label='rvalue_target')
plt.legend()
plt.show()
