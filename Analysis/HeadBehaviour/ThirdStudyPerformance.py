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
    for roll in rolling_size:
        try:

            print(subject, env, target, block, roll)

            output['rolling_average' + str(roll)] = get_new_angular_distance(
                output.head_rotation_y.rolling(roll, min_periods=1).mean(),
                output.head_rotation_x.rolling(roll, min_periods=1).mean(), output)

            result = dwell_analysis(target, env, block, subject, output, 'rolling_average' + str(roll))

            if result is not None:
                final_result.append(result)
        except Exception as e:
            print(e)
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
summary = pd.DataFrame(final_result)
summary.to_csv('summary3_rolling_average.csv')
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

#
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
rolling_mean_dataframe = summary_rolling_average.groupby(['environment', 'apply']).mean()
rolling_mean_dataframe = rolling_mean_dataframe.reindex(natsorted(rolling_mean_dataframe.index))
rolling_mean_dataframe = pd.concat([default_mean_dataframe, rolling_mean_dataframe])
summary_lowpass = pd.read_csv('summary3_lowpass.csv')
lowpass_mean_dataframe = summary_lowpass.groupby(['environment', 'apply']).mean()
lowpass_mean_dataframe = lowpass_mean_dataframe.reindex(natsorted(lowpass_mean_dataframe.index))


# %%
def find_int_string(string):
    if string == 'default': return string
    return "".join(filter(str.isdigit, string))


plot_columns = ['initial_contact_time', 'max_angle_distance', 'target_on_rate', 'target_in_count',
                'mean_target_on_time', 'longets_target_on_time']
# non-dwell-wise outcomes
# for col in plot_columns:
#     xtick_labels = pd.Series(rolling_mean_dataframe.loc['U'].index).apply(find_int_string)
#     ind = np.arange(len(xtick_labels))
#     width = 0.3
#     plt.bar(ind, rolling_mean_dataframe.loc['U'][col], width=width, alpha=0.75, label='UI')
#     plt.bar(ind + width, rolling_mean_dataframe.loc['W'][col], width=width, alpha=0.75, label='World')
#     # rolling_mean_dataframe.loc['U']['initial_contact_time'].plot()
#     # rolling_mean_dataframe.loc['W']['initial_contact_time'].plot()
#     plt.xticks(ind + width, xtick_labels)
#     plt.xlabel('rolling window size (frame)')
#     plt.ylabel(col)
#     plt.title(col)
#     plt.legend()
#     plt.show()
# dwell-wise outcomes
dwell_plot_columns = ['dwell_prior_count', 'dwell_success_count', 'first_dwell']
for col in dwell_plot_columns:
    fig,ax = plt.subplots(figsize=(10,10))
    columns = [column for column in rolling_mean_dataframe.columns if col in column]
    d = rolling_mean_dataframe.loc['W'][columns]
    sns.heatmap(d,cmap='Blues',annot=True)
    plt.title(col)
    plt.show()


    # for row in d.iterrows():
    #     plt.plot(pd.Series(d.index).apply(find_int_string), d.values)
    # plt.title(col)
    # plt.show()
