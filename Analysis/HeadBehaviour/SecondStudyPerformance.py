# %%
from AnalysingFunctions import *

from FileHandling import *


dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


# %%


# second study: subject= 1~11 , env= U,W , pos = S,W , target= 0~7 , block= 0~5
# third study : subject = 1~16 , env = U,W, pos = None, targets = 0~7, block = 0~5

# second
# subjects = range(1, 12)
# envs = ["U", "W"]
# poss = ['S', 'W']
# targets = range(8)
# blocks = range(1, 5)
#
# final_result = []
#
# for subject, env, pos, target, block in itertools.product(
#         subjects, envs, poss, targets, blocks
# ):
#     try:
#         print(subject, env, pos, target, block)
#         output = read_hololens_data(target=target, environment=env, posture=pos, block=block, subject=subject,
#                                     study_num=2)
#
#         walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
#         if (walklength < 3.5) and (pos == 'W'):
#             print('too short walklength');
#             continue;
#
#         if env == 'W':
#             r = 0.05 * 5
#         else:
#             r = 0.05
#         output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
#         initial_contact_time = output[output['angular_distance'] < output['MaximumTargetAngle']].timestamp.values[0]
#
#         # print('initial contact:', initial_contact_time)
#         Approach = output[output.timestamp <= initial_contact_time].reset_index(drop=True)
#         Maintain = output[output.timestamp > initial_contact_time].reset_index(drop=True)
#         Maintain['targetOn'] = np.where(Maintain['angular_distance'] < Maintain['MaximumTargetAngle'], True, False)
#         mean_error = Maintain.TargetHorizontal.mean()
#         std_error = Maintain.TargetHorizontal.std()
#         # print('mean error:', mean_error, 'std error', std_error)
#         dwell_list = []
#         maintain_total_frame = Maintain.shape[0]
#         target_on_frame = (Maintain[Maintain.targetOn == True]).shape[0]
#
#         # for k, g in itertools.groupby(Maintain.targetOn):  # search consecutive True (target-on) data
#         #     if k:  # if target-on
#         #         g = list(g)
#         #         dwell_list.append(g)
#         for k, l in itertools.groupby(Maintain.iterrows(), key=lambda row: row[1]['targetOn']):
#             dwell_list.append(pd.DataFrame([t[1] for t in l]))
#
#         # print('total frame:', maintain_total_frame, 'target-on frame:', target_on_frame, 'target-on rate',
#         #       target_on_frame / maintain_total_frame * 100, '%')
#         target_in_count = len(dwell_list)
#         # print('target in count:', target_in_count)
#
#         result = dict(
#             target=target, environment=env, posture=pos, block=block, subject=subject,
#             walklength=walklength,
#             initial_contact_time=initial_contact_time,
#             mean_error=mean_error, std_error=std_error,
#             maintain_total_frame=maintain_total_frame, target_on_frame=target_on_frame,
#             target_on_rate=target_on_frame / maintain_total_frame * 100,
#             target_in_count=target_in_count,
#         )
#
#         for threshold in dwell_thresholds:
#             dwell_success_list = []
#             for dwell in dwell_list:
#                 if len(dwell) * 1 / 60 > threshold:
#                     dwell_success_list.append(dwell)
#             if len(dwell_success_list) <= 0:
#                 print('no dwell success');
#                 continue;
#             first_dwell_success_time = dwell_success_list[0].timestamp.iloc[0]
#
#             dwell_success_frame = 0
#             mean_angular_speed = []
#
#             for success_dwell in dwell_success_list:  # loop through the successive dwells
#                 mean_angular_speed.append(success_dwell.angle_speed.mean())
#
#                 dwell_success_frame += len(success_dwell)
#             result['dwell_success_frame_' + str(threshold)] = dwell_success_frame
#             result['dwell_success_count_' + str(threshold)] = len(dwell_success_list)
#             result['first_dwell_' + str(threshold)] = first_dwell_success_time
#             result['mean_angular_speed_' + str(threshold)] = sum(mean_angular_speed) / len(dwell_success_list)
#             # print('threshold', threshold, 'dwell success frame', dwell_success_frame,
#             #       'dwell success count', len(dwell_success_list), 'first dwell', first_dwell_success_time,
#             #       'mean angular speed',
#             #       sum(mean_angular_speed) / len(dwell_success_list))
#
#         rolling_size = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#         for window in rolling_size:
#             t, m, sd = compare_horizontal(output.head_rotation_y.rolling(window, min_periods=1).mean(), output)
#             result['H_rolling_initial_contact_time_' + str(window)] = t
#             result['H_rolling_mean_error_' + str(window)] = m
#             result['H_rolling_std_error_' + str(window)] = sd
#             t, m, sd = compare_vertical(output.head_rotation_x.rolling(window, min_periods=1).mean(), output)
#             result['V_rolling_initial_contact_time_' + str(window)] = t
#             result['V_rolling_mean_error_' + str(window)] = m
#             result['V_rolling_std_error_' + str(window)] = sd
#         # lowpass_cutoff = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#         lowpass_cutoff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         for cutoff in lowpass_cutoff:
#             t, m, sd = compare_horizontal(butter_lowpass_filter(output.head_rotation_y, cutoff, 60, 2), output)
#             result['H_lowpass_initial_contact_time_' + str(cutoff)] = t
#             result['H_lowpass_mean_error_' + str(cutoff)] = m
#             result['H_lowpass_std_error_' + str(cutoff)] = sd
#             t, m, sd = compare_vertical(butter_lowpass_filter(output.head_rotation_x, cutoff, 60, 2), output)
#             result['V_lowpass_initial_contact_time_' + str(cutoff)] = t
#             result['V_lowpass_mean_error_' + str(cutoff)] = m
#             result['V_lowpass_std_error_' + str(cutoff)] = sd
#
#         final_result.append(result)
#     except Exception as e:
#         print(e)
# summary = pd.DataFrame(final_result)
# summary.to_csv('summary2.csv')
# %% second
summary = pd.read_csv('summary2.csv')
stand = summary[summary.posture == 'S']
walk = summary[(summary.posture == 'W') & (summary.walklength > 3.5)]
stand_UI = stand[stand.environment == 'U']
stand_World = stand[stand.environment == 'W']
walk_UI = walk[walk.environment == 'U']
walk_World = walk[walk.environment == 'W']

stand_UI0=stand_UI.fillna(0)
stand_World0=stand_World.fillna(0)
walk_UI0= walk_UI.fillna(0)
walk_World0=walk_World.fillna(0)



# plot_columns = ['initial_contact_time', 'mean_error', 'std_error', 'target_on_rate', 'target_in_count']
# for columnName in plot_columns:
#     draw_simple_plot(columnName)
dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
plot_dwell_columns = ['dwell_success_frame', 'dwell_success_count',  'mean_angular_speed']
for columnName in plot_dwell_columns:

    draw_stand_walk_comparison_plot(stand_UI0, stand_World0, walk_UI0, walk_World0, columnName, dwell_thresholds)
plot_dwell_columns_all=['first_dwell']
for columnName in plot_dwell_columns_all:
    draw_stand_walk_comparison_plot(stand_UI, stand_World, walk_UI, walk_World, columnName, dwell_thresholds)

rolling_size=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
plot_rolling_columns = ['H_rolling_initial_contact_time', 'H_rolling_std_error',
                        'V_rolling_initial_contact_time', 'V_rolling_std_error']
for columnName in plot_rolling_columns:
    # draw_rolling_plot(columnName)
    draw_stand_walk_comparison_plot(stand_UI, stand_World, walk_UI, walk_World, columnName, rolling_size)

lowpass_cutoff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plot_lowpass_columns = ['H_lowpass_initial_contact_time', 'H_lowpass_std_error',
                        'V_lowpass_initial_contact_time', 'V_lowpass_std_error']
for columnName in plot_lowpass_columns:
    draw_stand_walk_comparison_plot(stand_UI,stand_World,walk_UI,walk_World,columnName, lowpass_cutoff)

#%% fail/success
def show_success_rate(data,name:str):
    fail_count = data.isnull().sum(axis=0)
    total_trial = len(data)
    print(len(walk_UI))
    success_rate=[]
    for th in dwell_thresholds:
        success_rate.append((total_trial - fail_count['first_dwell_'+str(th)])/total_trial * 100)
    fig, ax = plt.subplots()
    # rect = ax.bar(simple_x, height, yerr=yerr, capsize=10)
    # width=0.1
    ax.scatter(dwell_thresholds,success_rate)
    ax.set_xlabel('dwell threshold (sec)')
    ax.set_ylabel('success rate (%)')
    ax.set_title(name)
    plt.show()

show_success_rate(walk_UI,'walk UI')
show_success_rate(walk_World,'walk World')
