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
subjects = [2]
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
    print(subject, env, target, block, 'mean eye confidence', eye.confidence.mean())
    if eye.confidence.mean() < 0.80:
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
    Veye1 = interpolate.interp1d(eye.timestamp, eye.norm_y, fill_value='extrapolate')
    Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
    Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
    Heye = interpolate.interp1d(eye.timestamp, eye.phi, fill_value='extrapolate')
    Heye1 = interpolate.interp1d(eye.timestamp, eye.norm_x, fill_value='extrapolate')
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
    Heye1 = pd.Series(Heye1(Timestamp))
    Veye1 = pd.Series(Veye1(Timestamp))
    Heye = Heye * 180 / math.pi
    Heye = Heye - Heye[0]
    Veye = Veye * 180 / math.pi
    Veye = Veye - Veye[0]
    from scipy.signal import medfilt

    for i in [601]:  # median filter
        Heye_ = (pd.Series([Heye[0]] * i)).append(Heye).append(pd.Series([Heye.iloc[-1]] * i))
        Veye_ = (pd.Series([Veye[0]] * i)).append(Veye).append(pd.Series([Veye.iloc[-1]] * i))
        a = medfilt(Heye_, i)
        b = medfilt(Veye_, i)
        Hmed_eye = a[i:-i]
        Vmed_eye = b[i:-i]

    fc = 3
    w = fc / (200 / 2)
    b, a = scipy.signal.butter(2, w, 'low')
    HMaintain_offset = signal.filtfilt(b, a, (Hholo - HTarget)[index:])
    HMaintain_simple = signal.filtfilt(b, a, (Hholo - HTarget + Heye - Heye[0])[index:])
    HMaintain_median = signal.filtfilt(b, a, (Heye - Hmed_eye + Hholo - HTarget)[index:])
    VMaintain_offset = signal.filtfilt(b, a, (Vholo - VTarget)[index:])
    VMaintain_simple = signal.filtfilt(b, a, (Vholo - VTarget + Veye - Veye[0])[index:])
    VMaintain_median = signal.filtfilt(b, a, (Veye - Vmed_eye + Vholo - VTarget)[index:])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax[0, 0].plot(Timestamp, 200 * (Heye1 - Heye1[0]), alpha=0.3, color='cyan')
    ax[0, 0].plot(Timestamp, Hholo - HTarget, alpha=0.3, color='blue')
    ax[0, 0].plot(Timestamp, Heye - Heye[0], alpha=0.3, color='red')
    ax[0, 0].plot(Timestamp, Heye - Hmed_eye, alpha=0.3, color='green')
    ax[0, 0].plot(Timestamp, signal.filtfilt(b, a, Hholo - HTarget), alpha=1.0, color='blue', label='head-offset')
    ax[0, 0].plot(Timestamp, signal.filtfilt(b, a, Heye - Heye[0]), alpha=1.0, color='red', label='raw eye')
    ax[0, 0].plot(Timestamp, signal.filtfilt(b, a, Heye - Hmed_eye), alpha=1.0, color='green',
                  label='median-filtered eye')
    ax[0, 0].set_title("Horizontal")
    ax[0, 0].axvline(initial_contact_time)
    ax[0, 0].axhline(0)
    ax[0, 0].set_ylim(-4, 4)

    ax[0, 1].plot(Timestamp, 200 * (Veye1 - Veye1[0]), alpha=0.3, color='cyan')
    ax[0, 1].plot(Timestamp, Vholo - VTarget, alpha=0.3, color='blue')
    ax[0, 1].plot(Timestamp, Veye - Veye[0], alpha=0.3, color='red')
    ax[0, 1].plot(Timestamp, Veye - Vmed_eye, alpha=0.3, color='green')
    ax[0, 1].plot(Timestamp, signal.filtfilt(b, a, Vholo - VTarget), alpha=1.0, color='blue', label='head-offset')
    ax[0, 1].plot(Timestamp, signal.filtfilt(b, a, Veye - Veye[0]), alpha=1.0, color='red', label='raw eye')
    ax[0, 1].plot(Timestamp, signal.filtfilt(b, a, Veye - Vmed_eye), alpha=1.0, color='green',
                  label='median-filtered eye')
    ax[0, 1].set_title("Vertical")
    ax[0, 1].axvline(initial_contact_time)
    ax[0, 1].axhline(0)
    ax[0, 1].set_ylim(-4, 4)
    ax[0, 1].legend()

    ax[1, 0].plot(Timestamp[index:],HMaintain_offset, label='offset', color='blue')
    ax[1, 0].plot(Timestamp[index:],HMaintain_simple, label='simple', color='red')
    ax[1, 0].plot(Timestamp[index:],HMaintain_median, label='median', color='green')
    ax[1, 0].axhline(0)

    ax[1, 1].plot(Timestamp[index:],VMaintain_offset, label='offset', color='blue')
    ax[1, 1].plot(Timestamp[index:],VMaintain_simple, label='simple', color='red')
    ax[1, 1].plot(Timestamp[index:],VMaintain_median, label='median', color='green')
    ax[1, 1].axhline(0)
    plt.legend()
    fig.suptitle('S' + str(subject) + " E" + str(env) + " T" + str(target) + " B" + str(block))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(HMaintain_offset, VMaintain_offset, marker='+', label='offset')
    ax.scatter(HMaintain_median, VMaintain_median, marker='+', label='median')
    confidence_ellipse(HMaintain_offset, VMaintain_offset, ax, edgecolor='blue', n_std=3)
    confidence_ellipse(HMaintain_median, VMaintain_median, ax, edgecolor='red', n_std=3)
    plt.xlim(-2, 2);
    plt.ylim(-2, 2)
    plt.axvline(0);
    plt.axhline(0)
    plt.legend()
    plt.title('S' + str(subject) + " E" + str(env) + " T" + str(target) + " B" + str(block))
    plt.show()
    data = dict(
        subject=subject, env=env, target=target, block=block,
        Hoffset_mean=HMaintain_offset.mean(), Hsimple_mean=HMaintain_simple.mean(),
        Hmedian_mean=HMaintain_median.mean(),
        Hoffset_std=HMaintain_offset.std(), Hsimple_std=HMaintain_simple.std(), Hmedian_std=HMaintain_median.std(),
        Voffset_mean=VMaintain_offset.mean(), Vsimple_mean=VMaintain_simple.mean(),
        Vmedian_mean=VMaintain_median.mean(),
        Voffset_std=VMaintain_offset.std(), Vsimple_std=VMaintain_simple.std(), Vmedian_std=VMaintain_median.std(),
    )
    # ETE(entring target event)
    for size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        count_offset = 0
        count_simple = 0
        count_median = 0
        time_offset = 0
        time_simple = 0
        time_median = 0
        for k, g in itertools.groupby(HMaintain_offset, lambda x: abs(x) <= size):
            if k == True: count_offset += 1;time_offset += len(list(g)) / 200
        for k, g in itertools.groupby(HMaintain_simple, lambda x: abs(x) <= size):
            if k == True: count_simple += 1;time_simple += len(list(g)) / 200
        for k, g in itertools.groupby(HMaintain_median, lambda x: abs(x) <= size):
            if k == True: count_median += 1;time_median += len(list(g)) / 200
        # print(size, count_offset, count_simple, count_median)
        data.update({
            str(size) + 'offset_ETE': count_offset,
            str(size) + 'simple_ETE': count_simple,
            str(size) + 'median_ETE': count_median,
            str(size) + 'offset_dwelltime': time_offset,
            str(size) + 'simple_dwelltime': time_simple,
            str(size) + 'median_dwelltime': time_median,
        })
    final_df.append(data)
final_df = pd.DataFrame(final_df)
final_df.to_csv("ETE.csv")
# %%
ete = pd.read_csv("ETE.csv")
sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for column in ['_ETE', '_dwelltime']:
    fig = go.Figure(data=[
        go.Bar(name='Offset', x=sizes, y=[ete[str(x) + 'offset' + column].mean() for x in sizes]),
        go.Bar(name='Simple', x=sizes, y=[ete[str(x) + 'simple' + column].mean() for x in sizes]),
        go.Bar(name='Median', x=sizes, y=[ete[str(x) + 'median' + column].mean() for x in sizes])
    ])
    fig.update_layout(barmode='group')
    fig.show()
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
        highfc = 0.1
        highw = highfc / (200 / 2)
        highb, higha = scipy.signal.butter(2, highw, 'high')
        row['Heye'] = pd.Series(signal.filtfilt(b, a, row['Heye']))
        zi = signal.lfilter_zi(highb, higha) * row['Heye'][0]

        lfilter_Heye = []

        for x in list(row['Heye']):
            y, zi = signal.lfilter(highb, higha, [x], zi=zi)
            lfilter_Heye.append(y[0])

        t = row['Hholo'] - row['HTarget']
        lfilter_Heye = pd.Series(lfilter_Heye)
        filtfilt_Heye = signal.filtfilt(hb, ha, row['Heye'])
        mean_Heye = row['Heye'] - row['Heye'][0]
        # roll_Heye=row['Heye']-row['Heye'].rolling(400,min_periods=1).mean()
        roll_Heye = signal.detrend(row['Heye'])
        plt.plot(-t, color='blue', label='target')
        # plt.plot(lfilter_Heye,color='orange',label='lfilter')
        # plt.plot(filtfilt_Heye,color='green',label='filtfilt')
        plt.plot(mean_Heye, color='red', label='mean')
        plt.plot(roll_Heye, color='pink', label='roll')
        plt.plot(t + mean_Heye, color='cyan')
        plt.plot(t + roll_Heye, color='gray')
        # plt.plot(signal.filtfilt(b,a,t+mean_Heye),color='cyan')
        plt.legend()
        plt.axvline(row['first_index'])
        plt.ylim(-3, 3)
        plt.show()
        # row['Himu'] =pd.Series( row['Himu'](np.arange(0,6.5,1/200)))
        # plt.scatter(row["Himu"].diff(1),pd.Series(row["Heye"]).diff(1))
        # # plt.plot(pd.Series(row["Heye"]).diff(1), label='eye vel')
        # plt.show()
        import plotly.figure_factory as ff

        hist_data = [t[row['first_index']:], (t + lfilter_Heye)[row['first_index']:],
                     (t + filtfilt_Heye)[row['first_index']:],
                     (t + row['Heye'] - row['Heye'].mean())[row['first_index']:]]
        labels = ['target', 'lfilter_Heye', 'filtfilt_Heye', 'comp']
        fig = ff.create_distplot(hist_data, labels, bin_size=.01)
        fig.update_layout(title="t:" + str(hist_data[0].std()) + " ,Eye: " + str(hist_data[1].std()) + " ,comp: " + str(
            hist_data[2].std()) + " hp cut:" + str(fch))
        fig.show()
        print(row['first_index'], row['target'])
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
