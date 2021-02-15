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
dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
rolling_size = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
cutoff_freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def inverse(a):
    return 1 / a


def vector_analysis(target, env, block, subject):
    output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                study_num=3)
    eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    print('mean eye confidence', eye.confidence.mean())
    if eye.confidence.mean() < 0.8:
        print('Too low eye confidence')
        return
    eye = eye[eye['confidence'] > 0.8]
    imu = read_imu_data(target=target, environment=env, posture='W', block=block, subject=subject,
                        study_num=3)
    shift, corr, shift_time = synchronise_timestamp(imu, output, show_plot=False)
    eye.timestamp = eye.timestamp - shift_time
    imu.timestamp = imu.timestamp - shift_time
    Timestamp = np.arange(0, 6.5, 1 / 120)
    Vholo = interpolate.interp1d(output.timestamp, output.head_rotation_x, fill_value='extrapolate')
    Vimu = interpolate.interp1d(imu.timestamp, imu.rotationX, fill_value='extrapolate')
    Veye = interpolate.interp1d(eye.timestamp, eye.norm_y, fill_value='extrapolate')
    Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
    Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
    Heye = interpolate.interp1d(eye.timestamp, eye.norm_x, fill_value='extrapolate')
    AngleSpeed = interpolate.interp1d(output.timestamp, output.angle_speed, fill_value='extrapolate')
    Hpre_cutoff = 5.0
    Vpre_cutoff = 5.0

    walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
    if (walklength < 3.5):
        print('too short walklength');
        return
        # continue;

    if env == 'W':
        r = 0.3 / 2
    else:
        r = 0.3 / 2
    apply = 'angular_distance'
    output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
    initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
    if len(initial_contact_time_data) <= 0:
        return;
    initial_contact_time = initial_contact_time_data.timestamp.values[0]

    Vholo = pd.Series(Vholo(Timestamp))
    Vimu = pd.Series(realtime_lowpass(Timestamp, Vimu(Timestamp), Vpre_cutoff))
    Veye = pd.Series(realtime_lowpass(Timestamp, Veye(Timestamp), Vpre_cutoff))
    Hholo = pd.Series(Hholo(Timestamp))
    Himu = pd.Series(realtime_lowpass(Timestamp, Himu(Timestamp), Hpre_cutoff))
    Heye = pd.Series(realtime_lowpass(Timestamp, Heye(Timestamp), Hpre_cutoff))
    vector = (Heye.diff(1) * Himu.diff(1) + Veye.diff(1) * Vimu.diff(1))
    index = len(Timestamp[Timestamp < initial_contact_time])
    vector_df = get_angle_between_vectors(Heye.diff(1), Himu.diff(1), Veye.diff(1), Vimu.diff(1))
    # APPROACH

    # MAINTAIN
    return index, vector_df, vector[:index], vector[index:]

    # Approach = output[output.timestamp <= initial_contact_time].reset_index(drop=True)
    # Maintain = output[output.timestamp > initial_contact_time].reset_index(drop=True)

    # Maintain['targetOn'] = np.where(Maintain[apply] < Maintain['MaximumTargetAngle'], True, False)


def dwell_analysis(target, env, block, subject, output, apply: str):
    walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
    if (walklength < 3.5):
        print('too short walklength');
        return
        # continue;

    if env == 'W':
        r = 0.3 / 2
    else:
        r = 0.3 / 2
    # output['MaximumTargetAngle'] = output.Distance.apply(inverse) * r
    # output['MaximumTargetAngle'] = output['MaximumTargetAngle'].apply(math.asin)
    output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
    # output['MaximumTargetAngle'] =output['MaximumTargetAngle'].apply(math.degrees)

    # plt.plot(output.timestamp,(r*1/output.Distance).apply(math.asin) * 180 / math.pi)
    # plt.plot(output.timestamp,output.MaximumTargetAngle)
    # plt.show()
    # plt.plot(output.timestamp,output[apply])
    # plt.show()
    # if apply == 'default':
    #     apply = 'angular_distance'
    initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
    if len(initial_contact_time_data) <= 0:
        return;
    initial_contact_time = initial_contact_time_data.timestamp.values[0]

    # print('initial contact:', initial_contact_time)
    Approach = output[output.timestamp <= initial_contact_time].reset_index(drop=True)
    Maintain = output[output.timestamp > initial_contact_time].reset_index(drop=True)
    Maintain['targetOn'] = np.where(Maintain[apply] < Maintain['MaximumTargetAngle'], True, False)
    mean_error = Maintain.TargetHorizontal.mean()
    std_error = Maintain.TargetHorizontal.std()
    max_angle_distance = Maintain[apply].max()
    mean_angle_distance = Maintain[apply].mean()

    # plt.plot(Maintain.targetOn)
    # plt.show()
    # plt.plot(Maintain.target_entered)
    # plt.show()
    # print('mean error:', mean_error, 'std error', std_error)

    dwell_list = []
    maintain_total_frame = Maintain.shape[0]
    target_on_frame = len(Maintain[Maintain.targetOn == True])

    for k, l in itertools.groupby(Maintain.iterrows(), key=lambda row: row[1]['targetOn']):
        if k == True:
            dwell_list.append(pd.DataFrame([t[1] for t in l]))

    # print('total frame:', maintain_total_frame, 'target-on frame:', target_on_frame, 'target-on rate',
    #       target_on_frame / maintain_total_frame * 100, '%')

    target_in_count = len(dwell_list)
    # print('target in count:', target_in_count)

    if target_in_count > 0:
        total_target_on_time = sum([dwell.timestamp.values[-1] - dwell.timestamp.values[0] for dwell in dwell_list])
        longets_target_on_time = max([dwell.timestamp.values[-1] - dwell.timestamp.values[0] for dwell in dwell_list])
        mean_target_on_time = total_target_on_time / target_in_count
    else:  # if there is no target-in
        total_target_on_time = 0
        mean_target_on_time = 0
        longets_target_on_time = 0
    result = dict(apply=apply,
                  target=target, environment=env, block=block, subject=subject,
                  walklength=walklength,
                  initial_contact_time=initial_contact_time,
                  mean_error=mean_error, std_error=std_error, max_angle_distance=max_angle_distance,
                  mean_angle_distance=mean_angle_distance,
                  maintain_total_frame=maintain_total_frame, target_on_frame=target_on_frame,
                  target_on_rate=target_on_frame / maintain_total_frame * 100,
                  target_in_count=target_in_count, total_target_on_time=total_target_on_time,
                  mean_target_on_time=mean_target_on_time, longets_target_on_time=longets_target_on_time
                  )

    for threshold in dwell_thresholds:
        dwell_success_list = []
        prior_count = 0
        prior_go = True
        for dwell in dwell_list:

            if len(dwell) * 1 / 60 > threshold:
                prior_go = False
                dwell_success_list.append(dwell)
            if prior_go == True:
                prior_count += 1

        if len(dwell_success_list) <= 0:
            # print('no dwell success');
            continue;
        first_dwell_success_time = dwell_success_list[0].timestamp.iloc[0]
        # print('first dwell success', first_dwell_success_time)
        dwell_success_frame = 0
        mean_angular_speed = []

        for success_dwell in dwell_success_list:  # loop through the successive dwells
            mean_angular_speed.append(success_dwell.angle_speed.mean())

            dwell_success_frame += len(success_dwell)
        result['dwell_prior_count_' + str(threshold)] = prior_count
        result['dwell_success_frame_' + str(threshold)] = dwell_success_frame
        result['dwell_success_count_' + str(threshold)] = len(dwell_success_list)
        result['first_dwell_' + str(threshold)] = first_dwell_success_time
        result['mean_angular_speed_' + str(threshold)] = sum(mean_angular_speed) / len(dwell_success_list)

    return result


# %%
# third

# subjects = [1]
# envs = ['W']
# targets = [0]
# blocks = [1]

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
    output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                study_num=3)
    # for roll in rolling_size:
    #     try:
    #
    #         print(subject, env, target, block, roll)
    #
    #         output['rolling_average' + str(roll)] = get_new_angular_distance(
    #             output.head_rotation_y.rolling(roll, min_periods=1).mean(),
    #             output.head_rotation_x.rolling(roll, min_periods=1).mean(), output)
    #
    #         result = dwell_analysis(target, env, block, subject, output, 'rolling_average' + str(roll))
    #
    #         if result is not None:
    #             final_result.append(result)
    #     except Exception as e:
    #         print(e)
    # for cutoff in cutoff_freqs:
    #     try:
    #
    #         print(subject, env, target, block, cutoff)
    #
    #         output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
    #                                     study_num=3)
    #         output['lowpass' + str(cutoff)] = get_new_angular_distance(
    #             pd.Series(realtime_lowpass(output.timestamp,output.head_rotation_y,cutoff)),
    #             pd.Series(realtime_lowpass(output.timestamp,output.head_rotation_x,cutoff)), output)
    #
    #         result = dwell_analysis(target, env, block, subject, output, 'lowpass' + str(cutoff))
    #
    #         if result is not None:
    #             final_result.append(result)
    #     except Exception as e:
    #         print(e)
    # try:
    #     print(subject, env, target, block)
    #     output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
    #                                 study_num=3)
    #     output['default'] = get_new_angular_distance(
    #         output.head_rotation_y, output.head_rotation_x, output
    #     )
    #     result = dwell_analysis(target, env, block, subject, output, 'default')
    #
    #     if result is not None:
    #         final_result.append(result)
    # except Exception as e:
    #     print(e)

    # betas = [1, 0.1, 0.01, 0.001, 0.0001]
    betas = [0.1, 0.01, 0.001]
    for cutoff in cutoff_freqs:
        for beta in betas:
            print(subject, env, target, block, cutoff, beta)
            try:
                output['one_euro' + str(cutoff) + '_' + str(beta)] = get_new_angular_distance(
                    pd.Series(one_euro(output.head_rotation_y, output.timestamp, 60, cutoff, beta)),
                    pd.Series(one_euro(output.head_rotation_x, output.timestamp, 60, cutoff, beta)),
                    output)

                result = dwell_analysis(target, env, block, subject, output, 'one_euro' + str(cutoff) + '_' + str(beta))

                if result is not None:
                    final_result.append(result)
            except Exception as e:
                print(e)
summary = pd.DataFrame(final_result)
summary.to_csv('summary3_oneEuro_meanAngleDistance.csv')
print('overall time', time.time() - t)
# %%


summary = pd.read_csv('summary3.csv')

UI = summary[summary.environment == 'U']
World = summary[summary.environment == 'W']
comparison = ['default'] + ['rolling_average' + str(roll) for roll in rolling_size] + ['lowpass' + str(cutoff) for
                                                                                       cutoff in cutoff_freqs]

mean_dataframe = summary.groupby(['environment', 'apply']).mean()
mean_dataframe = mean_dataframe.reindex(natsorted(mean_dataframe.index))

dwell_success_frame_columns = [col for col in mean_dataframe.columns if 'dwell_success_frame' in col]
dwell_success_count_columns = [col for col in mean_dataframe.columns if 'dwell_success_count' in col]

first_dwell_columns = [col for col in mean_dataframe.columns if 'first_dwell' in col]
# UI_dwell_success_rate = 100 - UI.isnull().groupby('apply').sum().astype(int) / len(UI)*100
UI_dwell_success_rate = UI.groupby('apply').apply(lambda x: x.notnull().mean())
UI_dwell_success_rate = UI_dwell_success_rate.reindex(natsorted(UI_dwell_success_rate.index))

UI_dwell_success_rate[dwell_success_frame_columns].plot()
plt.show()
mean_dataframe.loc['U'][dwell_success_count_columns].plot()
plt.show()
mean_dataframe.loc['U'][dwell_success_frame_columns].plot()
plt.show()
mean_dataframe.loc['U'][first_dwell_columns].plot()
plt.show()

# %%
fig_basic, (ax_init_time, ax_max_angle_distance, ax_rate, ax_in_count) = plt.subplots(4, 1, figsize=(10, 20),
                                                                                      sharex=True)
fig_dwell, (ax_dwell_success_count, ax_first_dwell, ax_dwell_success_rate) = plt.subplots(3, 1, figsize=(10, 15),
                                                                                          sharex=True)

for subset in comparison:
    # data = World[World['apply'] == subset]
    data = UI[UI['apply'] == subset]
    print(subset, len(data), end=':')
    # print('init', data['initial_contact_time'].mean(), '\t',
    #       'max', data['max_angle_distance'].mean(), '\t',
    #       'rate', data['target_on_rate'].mean(), '\t',
    #       'in count', data['target_in_count'].mean(), '\t',
    #       'total time', data['total_target_on_time'].mean(), '\t',
    #       'mean time', data['mean_target_on_time'].mean(), '\t', )

    ax_init_time.bar(subset, data['initial_contact_time'].mean(), color=['red'])
    ax_max_angle_distance.bar(subset, data['max_angle_distance'].mean())
    ax_rate.bar(subset, data['target_on_rate'].mean())

    ax_in_count.bar(subset, data['target_in_count'].mean())

    dwell_success_counts = []
    dwell_first_dwells = []
    dwell_success_rates = []
    fail_count = data.isnull().sum(axis=0)
    for dwell_th in dwell_thresholds:
        total_trial = len(data['dwell_success_count_' + str(dwell_th)])

        dwell_success_rates.append(
            (total_trial - fail_count['dwell_success_count_' + str(dwell_th)]) / total_trial * 100
        )
        dwell_success_counts.append(data['dwell_success_count_' + str(dwell_th)].mean())
        dwell_first_dwells.append(data['first_dwell_' + str(dwell_th)].mean())
    ax_dwell_success_count.plot(dwell_thresholds, dwell_success_counts, label=subset)
    ax_first_dwell.plot(dwell_thresholds, dwell_first_dwells)
    ax_dwell_success_rate.plot(dwell_thresholds, dwell_success_rates)
ax_init_time.set_ylabel('initial contact time (s)')
ax_init_time.title.set_text('initial contact time (s)')
ax_max_angle_distance.set_ylabel('maximum angle distance (degree)')
ax_max_angle_distance.title.set_text('maximum angle distance (degree)')
ax_rate.set_ylabel('target-on rate (%)')
ax_rate.title.set_text('target-on rate (%)')
ax_in_count.set_ylabel('target-in count')
ax_in_count.title.set_text('target-in count')
ax_dwell_success_count.set_ylabel('dwell success count')
ax_dwell_success_count.title.set_text('dwell success count')
ax_first_dwell.set_ylabel('first dwell time (s)')
ax_first_dwell.title.set_text('first dwell time (s)')
ax_dwell_success_rate.set_ylabel('dwell success rate (%)')
ax_dwell_success_rate.title.set_text('dwell success rate (%)')
ax_dwell_success_rate.grid()
fig_basic.show()
fig_dwell.legend()
fig_dwell.show()

# %%
# summary1 = pd.read_csv('summary3.csv')
# summary2 = pd.read_csv('summary3_LP.csv')
# default_dataframe = summary1[summary1['apply']=='default']
# lowpass_dataframe=  summary2[summary2['apply'].str.contains('lowpass')  ]
# rolling_average_dataframe = summary2[summary2['apply'].str.contains('rolling_average')]
# summary = pd.concat([summary1,summary2])
# %%
summary_default = pd.read_csv('summary3_default.csv')
default_mean_dataframe = summary_default.groupby(['environment', 'apply']).mean()
default_mean_dataframe = default_mean_dataframe.reindex(natsorted(default_mean_dataframe.index))

summary_rolling_average = pd.read_csv('summary3_rolling_average.csv')
summary_rolling_average = pd.concat([summary_rolling_average, summary_default])
rolling_mean_dataframe = summary_rolling_average.groupby(['environment', 'apply']).mean()
rolling_mean_dataframe = rolling_mean_dataframe.reindex(natsorted(rolling_mean_dataframe.index))
# rolling_mean_dataframe = pd.concat([default_mean_dataframe, rolling_mean_dataframe])
summary_lowpass = pd.read_csv('summary3_lowpass.csv')
summary_lowpass = pd.concat([summary_lowpass, summary_default])
# summary_lowpass = summary_lowpass.reindex(natsorted(summary_lowpass.index))
lowpass_mean_dataframe = summary_lowpass.groupby(['environment', 'apply']).mean()
lowpass_mean_dataframe = lowpass_mean_dataframe.reindex(natsorted(lowpass_mean_dataframe.index))

summary_oneeuro = pd.read_csv('summary3_oneEuro.csv')
summary_oneeuro = pd.concat([summary_default, summary_oneeuro])
oneeuro_mean_dataframe = summary_oneeuro.groupby(['environment', 'apply']).mean()
oneeuro_mean_dataframe = oneeuro_mean_dataframe.reindex(natsorted(oneeuro_mean_dataframe.index))

# lowpass_mean_dataframe=pd.concat([default_mean_dataframe,lowpass_mean_dataframe])

# %%
# col40 = [col for col in oneeuro_mean_dataframe.columns if '4.0' in col]

plot_columns = ['initial_contact_time', 'max_angle_distance', 'target_on_rate', 'target_in_count',
                'mean_target_on_time', 'longets_target_on_time']

betas = [0.1, 0.01, 0.001, 0.0001]
betas = [str(beta) for beta in betas]
barPlot = []
for col in plot_columns:
    barPlot = []
    for beta in betas:
        ind = oneeuro_mean_dataframe.loc['U'].index[
            (oneeuro_mean_dataframe.loc['U'].index.str.contains('_' + str(beta))) | (
                oneeuro_mean_dataframe.loc['U'].index.str.contains('default'))]
        # barPlot.append(go.Bar(
        #     name=str(beta),
        #     x=[0] + cutoff_freqs, y=oneeuro_mean_dataframe.loc['U'][col].loc[ind],
        #     textposition='auto'
        # ))
        barPlot.append(go.Scatter(
            name=str(beta),
            x=[0] + cutoff_freqs, y=oneeuro_mean_dataframe.loc['U'][col].loc[ind],
        ))

    fig = go.Figure(data=barPlot)
    # fig.update_layout(title_text=col, barmode='group',xaxis_tickangle=-45)
    fig.update_layout(title_text=col, xaxis_tickangle=-45)
    fig.show()


# %%
def find_int_string(string):
    if string == 'default': return string
    return "".join(filter(str.isdigit, string))


plot_columns = ['initial_contact_time', 'max_angle_distance', 'target_on_rate', 'target_in_count',
                'mean_target_on_time', 'longets_target_on_time']
# non-dwell-wise outcomes
for col in plot_columns:
    xtick_labels = pd.Series(rolling_mean_dataframe.loc['U'].index).apply(find_int_string)
    ind = np.arange(len(xtick_labels))
    width = 0.3
    plt.bar(ind, rolling_mean_dataframe.loc['U'][col], width=width, alpha=0.75, label='UI')
    plt.bar(ind + width, rolling_mean_dataframe.loc['W'][col], width=width, alpha=0.75, label='World')
    # rolling_mean_dataframe.loc['U']['initial_contact_time'].plot()
    # rolling_mean_dataframe.loc['W']['initial_contact_time'].plot()
    plt.xticks(ind + width, xtick_labels)
    plt.xlabel('rolling window size (frame)')
    plt.ylabel(col)
    plt.title(col)
    plt.legend()
    plt.show()

    xtick_labels = pd.Series(lowpass_mean_dataframe.loc['U'].index).apply(find_int_string)
    ind = np.arange(len(xtick_labels))
    width = 0.3
    plt.bar(ind, lowpass_mean_dataframe.loc['U'][col], width=width, alpha=0.75, label='UI')
    plt.bar(ind + width, lowpass_mean_dataframe.loc['W'][col], width=width, alpha=0.75, label='World')
    plt.xticks(ind + width, xtick_labels)
    plt.xlabel('lowpass cutoff frequency (Hz)')
    plt.ylabel(col)
    plt.title(col)
    plt.legend()
    plt.show()

# dwell-wise outcomes
dwell_plot_columns = ['dwell_prior_count', 'dwell_success_count', 'first_dwell']
for col in dwell_plot_columns:
    fig, ax = plt.subplots(figsize=(10, 10))
    columns = [column for column in rolling_mean_dataframe.columns if col in column]
    d = rolling_mean_dataframe.loc['U'][columns]
    sns.heatmap(d, cmap='Blues', annot=True)
    plt.title(col)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 10))
    columns = [column for column in lowpass_mean_dataframe.columns if col in column]
    d = lowpass_mean_dataframe.loc['U'][columns]
    sns.heatmap(d, cmap='Blues', annot=True)
    plt.title(col)
    plt.show()

    # for row in d.iterrows():
    #     plt.plot(pd.Series(d.index).apply(find_int_string), d.values)
    # plt.title(col)
    # plt.show()

# %%
# U_fail_count = summary_lowpass[summary_lowpass['environment']=='U'].isnull().sum(axis=0)
# W_fail_count = summary_lowpass[summary_lowpass['environment']=='W'].isnull().sum(axis=0)

# print(U_fail_count[columns]/len(summary_lowpass[summary_lowpass['environment']=='U'])*100)
#
# xtick_labels = pd.Series(U_fail_count[columns].index).apply(find_int_string)
# ind = np.arange(len(xtick_labels))
# width = 0.3
# plt.bar(ind, U_fail_count[columns].values, width=width, alpha=0.75, label='UI')
# plt.bar(ind + width, W_fail_count[columns].values, width=width, alpha=0.75, label='World')
# plt.xticks(ind + width, xtick_labels)
# plt.xlabel('lowpass cutoff frequency (Hz)')
# plt.ylabel(col)
# plt.title(col)
# plt.legend()
# plt.show()
# UI = summary_default[summary_default['environment']=='U']
# World = summary_default[summary_default['environment']=='W']
UI_LP = summary_lowpass[summary_lowpass['environment'] == 'U']
World_LP = summary_lowpass[summary_lowpass['environment'] == 'W']
UI_dwell_success_rate_LP = UI_LP.groupby('apply').apply(lambda x: x.notnull().mean())
World_dwell_success_rate_LP = World_LP.groupby('apply').apply(lambda x: x.notnull().mean())

UI_RA = summary_rolling_average[summary_rolling_average['environment'] == 'U']
World_RA = summary_rolling_average[summary_rolling_average['environment'] == 'W']
UI_dwell_success_rate_RA = UI_RA.groupby('apply').apply(lambda x: x.notnull().mean())
World_dwell_success_rate_RA = World_RA.groupby('apply').apply(lambda x: x.notnull().mean())

UI_dwell_success_rate_RA = UI_dwell_success_rate_RA.reindex(natsorted(UI_dwell_success_rate_RA.index))
World_dwell_success_rate_RA = World_dwell_success_rate_RA.reindex(natsorted(World_dwell_success_rate_RA.index))

UI_oneeuro = summary_oneeuro[summary_oneeuro['environment'] == 'U']
World_oneeuro = summary_oneeuro[summary_oneeuro['environment'] == 'W']
UI_dwell_success_rate_OE = UI_oneeuro.groupby('apply').apply(lambda x: x.notnull().mean())
World_dwell_success_rate_OE = World_oneeuro.groupby('apply').apply(lambda x: x.notnull().mean())

UI_dwell_success_rate_OE = UI_dwell_success_rate_OE.reindex(natsorted(UI_dwell_success_rate_OE.index))
World_dwell_success_rate_OE = World_dwell_success_rate_OE.reindex(natsorted(World_dwell_success_rate_OE.index))
# UI_dwell_success_rate_LP = UI_LP.notnull().sum(axis=0) / len(UI_LP) * 100
# a = UI_LP.groupby('apply')
# for g,k in a:
#
#     print(k.notnull().sum(axis=0)['first_dwell_1.2'])
# UI_dwell_success_rate_LP = UI_dwell_success_rate_LP.reindex(natsorted(UI_dwell_success_rate_LP.index))
# UI_dwell_success_rate = UI.groupby('apply').apply(lambda  x: x.notnull().mean())
# UI_dwell_success_rate = pd.concat([UI_dwell_success_rate,UI_dwell_success_rate_LP])
columns = [column for column in UI_RA.columns if 'first_dwell' in column]
# sns.heatmap(UI_dwell_success_rate_LP[columns])
# plt.title("UI-LP")
# plt.show()
# sns.heatmap(World_dwell_success_rate_LP[columns])
# plt.title("World-LP")
# plt.show()
# sns.heatmap(UI_dwell_success_rate_RA[columns])
# plt.title("UI-Rolling Average")
# plt.show()
# sns.heatmap(World_dwell_success_rate_RA[columns])
# plt.title("World-Rolling Average")
# plt.show()
# UI_dwell_success_rate_OE=pd.DataFrame(np.roll(UI_dwell_success_rate_OE.values,1,axis=0),
#                                       index=np.roll(UI_dwell_success_rate_OE.index,1),columns=UI_dwell_success_rate_OE.columns)
idx = UI_dwell_success_rate_OE.index.tolist()
idx.pop(0)
UI_dwell_success_rate_OE = UI_dwell_success_rate_OE.reindex(idx + ['default'])
sns.heatmap(UI_dwell_success_rate_OE[columns])
plt.show()
# %% heatmap in plotly way
import plotly.express as px

fig = px.imshow(UI_dwell_success_rate_OE[columns])

fig.show()

# %% watch only 500ms dwell
summary_whole = pd.concat([
    pd.read_csv('summary3_default.csv'),
    pd.read_csv('summary3_lowpass.csv'),
    pd.read_csv('summary3_oneEuro_meanAngleDistance.csv')
])
whole_mean_dataframe = summary_whole.groupby(['environment', 'apply']).mean()
whole_mean_dataframe = whole_mean_dataframe.reindex(natsorted(whole_mean_dataframe.index))
betas = [1, 0.1, 0.01, 0.001, 0.0001]
betas = [str(beta) for beta in betas]
filtered_plots = []
plot_env = 'U'
plot_dwell_threshold = 1.0
plot_data = whole_mean_dataframe.loc[plot_env]
dwell_plot_columns = ['dwell_prior_count', 'dwell_success_count', 'first_dwell', 'mean_angle_distance',
                      'initial_contact_time', 'max_angle_distance', 'target_on_rate', 'target_in_count',
                      'mean_target_on_time', 'longets_target_on_time'
                      ]
for col in dwell_plot_columns:
    filtered_plots = []
    columns = [column for column in plot_data.columns if (col + '_' + str(plot_dwell_threshold)) in column]
    if 'dwell' in col:
        col = col + '_' + str(plot_dwell_threshold)
    ind = plot_data.index[
        plot_data.index.str.contains('lowpass')
    ]
    filtered_plots.append(go.Bar(name='lowpass', x=cutoff_freqs, y=plot_data[col].loc[ind]))
    filtered_plots.append(go.Bar(name='default', x=['default'], y=[plot_data[col].loc['default']]))
    for beta in betas:
        ind = plot_data.index[
            plot_data.index.str.contains('_' + str(beta))
        ]
        filtered_plots.append(go.Bar(
            name=str(beta),
            x=cutoff_freqs, y=plot_data[col].loc[ind]
        ))
    fig = go.Figure(data=filtered_plots)
    fig.layout = go.Layout(xaxis=dict(type='category'))
    fig.update_layout(title_text=col, barmode='group')
    fig.show()

# %%
subject = 1
env = 'U'
target = 2
block = 4
index, vector_df, approach, maintain = vector_analysis(target, env, block, subject)
vector_df.eye_magnitude.plot()
plt.axvline(index);
plt.show()
vector_df.imu_magnitude.plot();
plt.axvline(index);
plt.show()

plt.scatter(vector_df.index, vector_df.angle, marker='+')
plt.axhline(y=90)
plt.axvline(x=vector_df.index[index])
plt.show()
# vector_df.angle.loc[:index].plot();plt.show()
sns.histplot(vector_df.angle.loc[:index], kde=True);
plt.show()
sns.histplot(vector_df.angle.loc[index:], kde=True);
plt.show()
# plt.plot(vector_df.index[1:],butter_lowpass_filter(vector_df.angle[1:],5,120,2,False))

# plt.fill_between(vector_df.index[1:],butter_lowpass_filter(vector_df.angle[1:],5,120,2,False),where=(butter_lowpass_filter(vector_df.angle[1:],5,120,2,False)<=90),facecolor='red',alpha=0.5)
# plt.show()
# approach.plot()
# maintain.plot()
# plt.show()
# %%
subject = 1
env = 'U'
target =6
block = 1
output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                            study_num=3)
eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                    study_num=3)
print('mean eye confidence', eye.confidence.mean())
if eye.confidence.mean() < 0.8:
    print('Too low eye confidence')
    # return
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
# Veye = interpolate.interp1d(eye.timestamp, eye.norm_y, fill_value='extrapolate')
Veye = interpolate.interp1d(eye.timestamp, -eye.theta, fill_value='extrapolate')
Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
# Heye = interpolate.interp1d(eye.timestamp, eye.norm_x, fill_value='extrapolate')
Heye = interpolate.interp1d(eye.timestamp, eye.phi, fill_value='extrapolate')
AngleSpeed = interpolate.interp1d(output.timestamp, output.angle_speed, fill_value='extrapolate')
Hpre_cutoff = 5.0
Vpre_cutoff = 5.0
walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
if (walklength < 3.5):
    print('too short walklength');

    # continue;

if env == 'W':
    r = 0.3 / 2
else:
    r = 0.3 / 2
apply = 'angular_distance'
output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
if len(initial_contact_time_data) <= 0:
    print('no contact')
initial_contact_time = initial_contact_time_data.timestamp.values[0]
VTarget = pd.Series(realtime_lowpass(Timestamp, VTarget(Timestamp), Vpre_cutoff))
Vholo = pd.Series(realtime_lowpass(Timestamp, Vholo(Timestamp), Vpre_cutoff))
Vimu = pd.Series(realtime_lowpass(Timestamp, Vimu(Timestamp), Vpre_cutoff))
Veye = pd.Series(realtime_lowpass(Timestamp, Veye(Timestamp), Vpre_cutoff))
HTarget = pd.Series(realtime_lowpass(Timestamp, HTarget(Timestamp), Hpre_cutoff))
Hholo = pd.Series(realtime_lowpass(Timestamp, Hholo(Timestamp), Hpre_cutoff))
Himu = pd.Series(realtime_lowpass(Timestamp, Himu(Timestamp), Hpre_cutoff))
Heye = pd.Series(realtime_lowpass(Timestamp, Heye(Timestamp), Hpre_cutoff))
Heye = Heye * 180/math.pi
Veye =Veye * 180/math.pi
vector = (Heye.diff(1) * Himu.diff(1) + Veye.diff(1) * Vimu.diff(1))
vector = vector * 200 * 200
index = len(Timestamp[Timestamp < initial_contact_time])
vector_df = get_angle_between_vectors(Heye.diff(1), Himu.diff(1), Veye.diff(1), Vimu.diff(1))
plt.plot(Himu)
# plt.show()
plt.plot(Himu[0] +Heye.diff(1)*200+Himu.diff(1)*200)
# plt.plot(Himu+Heye-Heye.mean())

plt.show()
# Hholo.plot();
# plt.axvline(index);plt.show()
# Himu.plot();
# plt.axvline(index);plt.show()
# Heye.plot();
# plt.axvline(index);plt.show()
# vector_df.eye_magnitude.plot()


# Heye.diff(1).plot()
# plt.axvline(index);plt.show()
# Veye.diff(1).plot()
# plt.axvline(index);plt.show()
# #
# vector_df.imu_magnitude.plot()
# plt.axvline(index);plt.show()
# Himu.plot()
# plt.axvline(index);plt.show()
# Vimu.plot()
# plt.axvline(index);plt.show()
# vector = vector.append(vector)

# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# lag = 20
#
# ax[0].plot(list(vector))
# # ax[0].axhline(5)
# ax[0].axvline(index, color='r');
# ax[1].axvline(index, color='r');
# # plt.show()
#
# # peak_detection = real_time_peak_detection(array=[0]*lag, lag=lag, threshold=5, influence=0.1)
# threshold = 10
# peak_detection = real_time_peak_detection(array=vector[:lag], lag=lag, threshold=threshold, influence=0.10)
# output = [0] * lag
# for n, i in enumerate(vector[lag:]):
#     # if i < 0: i = 0
#
#     p, avg, dev = peak_detection.thresholding_algo(i)
#     output.append(p)
#
#     # if p > 0.9 and i > 0:
#     #     ax[0].axvline(n + lag, alpha=0.2)
# # ax[0].plot(list(pd.Series(peak_detection.avgFilter)), color='cyan')
# # ax[0].plot(list(pd.Series(peak_detection.avgFilter) + pd.Series(peak_detection.stdFilter) * threshold), color='g')
# # ax[0].plot(list(pd.Series(peak_detection.avgFilter) - pd.Series(peak_detection.stdFilter) * threshold), color='g')
#
# # ax[0].plot(ii, upper, color='g')
# # ax[0].plot(ii, lower, color='g')
# ax[1].plot(output);
# for n, i in enumerate(Hholo.diff(1)):
#     if n == 0: continue
#     # if Hholo.diff(1).iloc[n]*Hholo.diff(1).iloc[n-1]<0 or Vholo.diff(1).iloc[n]*Vholo.diff(1).iloc[n-1]<0:
#     # ax[0].axvline(n,color='g')
# plt.show()
# ((Heye-Heye.mean())*180/math.pi).plot();
# (Hholo-Hholo.mean()).plot()
# ((Heye-Heye.mean())*180/math.pi+(Hholo-Hholo.mean())).plot();
# (HTarget-Hholo.mean()).plot()
# plt.show()
# # (Himu-Himu.mean()).plot();plt.show()
#
# ((Veye-Veye.mean())*180/math.pi).plot();
# (Vholo-Vholo.mean()).plot()
#
# ((Veye-Veye.mean())*180/math.pi+(Vholo-Vholo.mean())).plot();
# (VTarget-Vholo.mean()).plot()
# plt.show()
#
# ((Veye-Veye.mean())*180/math.pi).plot();
# (Vimu-Vimu.mean()).plot()
#
# ((Veye-Veye.mean())*180/math.pi+(Vimu-Vimu.mean())).plot();
# (VTarget-Vholo.mean()).plot()
# plt.show()
