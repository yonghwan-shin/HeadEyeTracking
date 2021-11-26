"""
cursor_angular_distance', 'start_num', 'end_num', 'timestamp',
'target_position_x', 'target_position_y', 'target_position_z',
'head_position_x', 'head_position_y', 'head_position_z',
'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
'head_forward_x', 'head_forward_y', 'head_forward_z', 'cursor_type',
'target_name', 'origin_x', 'origin_y', 'origin_z', 'direction_x',
'direction_y', 'direction_z', 'ray_origin_x', 'ray_origin_y',
'ray_origin_z', 'ray_direction_x', 'ray_direction_y',
'ray_direction_z']
"""
# %%
import itertools
import math

import numpy as np

import pandas as pd
from collections import defaultdict
from FileHandling import *

from AnalysisFunctions import *
from scipy.spatial.transform import Rotation as R

pio.renderers.default = 'browser'
pd.set_option('mode.chained_assignment', None)  # <==== 경고를 끈다
# %%


# %% test set

t = 5

#
# data = get_one_trial(3, 'WALK', 'EYE', 3, t)
data = read_hololens_data(7, 'WALK', 'EYE', t)
data['trial_check'] = data['end_num'].diff(1)
data['cursor_rotation'] = data.apply(
    lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
data['target_rotation'] = data.apply(
    lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                          x.target_position_z - x.origin_z), axis=1)
data['cursor_horizontal'] = data.apply(
    lambda x: math.sin(math.radians(x.cursor_rotation[1])), axis=1
)
data['cursor_vertical'] = data.apply(
    lambda x: x.cursor_rotation[0], axis=1
)
data['target_horizontal'] = data.apply(
    lambda x: math.sin(math.radians(x.target_rotation[1])), axis=1
)
data['target_vertical'] = data.apply(
    lambda x: x.target_rotation[0], axis=1
)
data['horizontal'] = data.apply(
    lambda x: change_angle(x.cursor_horizontal - x.target_horizontal), axis=1
)
data['vertical'] = data.apply(
    lambda x: change_angle(x.cursor_vertical - x.target_vertical), axis=1
)
data['target_vertical_velocity'] = (data['target_vertical'].diff(1) / data['timestamp'].diff(1)).rolling(30).mean()
data['target_horizontal_velocity'] = (data['target_horizontal'].diff(1) / data['timestamp'].diff(1))
data['target_horizontal_acc'] = (data['target_horizontal_velocity'].diff(1) / data['timestamp'].diff(1))
data['walking_speed'] = (
        (data['head_position_x'].diff(1).pow(2) + data['head_position_z'].diff(1).pow(2)).apply(math.sqrt) / data[
    'timestamp'].diff(1)).rolling(6, min_periods=1).mean()
data['walking_acc'] = (data['walking_speed'].diff(1) / data['timestamp'].diff(1))
data['target_speed'] = (
        (data['target_position_x'].diff(1).pow(2) + data['target_position_z'].diff(1).pow(2)).apply(math.sqrt) / data[
    'timestamp'].diff(1)).rolling(6, min_periods=1).mean()
# fig = px.histogram(data['target_horizontal_velocity'])
# fig.show()
# data.cursor_horizontal.plot()
# data.target_horizontal.plot()
# plt.show()
# data.cursor_vertical.plot()
# data.target_vertical.plot()
# plt.show()
# data.horizontal.plot()
# plt.show()
# data.vertical.plot()
# plt.show()
# data.walking_speed.plot()
# plt.plot(data.timestamp,data.walking_speed)
# plt.plot(data.timestamp,data.target_speed)
# plt.show()
#
# data['target_horizontal_velocity'].plot();
data_without_change = data[(data.trial_check == 0)][5:]
fail_check = data_without_change[(data_without_change.target_horizontal_velocity > 5) | (
        data_without_change.target_horizontal_velocity < -5)]
plt.plot(data_without_change.timestamp, data_without_change.target_horizontal_velocity)
#
plt.show()

#
# plt.plot(data.timestamp,data.walking_acc)
# plt.plot(data.timestamp,data.target_horizontal_acc)
#
# plt.show()


# %%
# subjects = range(24)
# total_horizontal_vels = []
# # total_horizontal_acc=[]
# # total_walking_acc=[]
# # total_walking_speeds = []
# for subject in subjects:
#     target_horizotal_vels, target_fail_count = discover_error(subject)
#     total_horizontal_vels += target_horizotal_vels
#     # total_horizontal_acc+=target_horizontal_acc
#     # total_walking_acc += walking_acc
#     # total_walking_speeds += walking_speeds
#     print(subject, target_fail_count)

# thv = pd.Series(total_horizontal_vels)
# fig = px.histogram(thv)
# fig.show()

# thv.describe()
#
# thv.quantile(q=0.995, interpolation='nearest')


# %%
# summary = pd.DataFrame(columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num'])


# %% extra_times_over_target
# subjects= [0,1,2,3,4,5,6,7,9]
# subjects = [9,10,11,12,13,14,15,16,17]
# subjects = [7]
subjects = range(24)
for subject in subjects:
    # a = summarize_subject(subject)
    a = target_size_analysis(subject)
dfs = []
for subject in subjects:
    dwell_summary = pd.read_csv('Target_size_Rawsummary' + str(subject) + '.csv')
    dfs.append(dwell_summary)
raw_df = pd.concat(dfs)
by_target_size = raw_df.groupby([raw_df.target_size, raw_df.posture, raw_df.cursor_type]).mean()
by_target_size.to_csv("target_size_summary_total.csv")
by_target_size = pd.read_csv("target_size_summary_total.csv")
fig = px.bar(by_target_size, x='cursor_type', y=['extra_times_over_target'], barmode='group',
             facet_row='posture', facet_col='target_size', title='')
fig.update_layout(title_text='extra_times_over_target')
fig.show()

# %% basic summary
subjects = range(24)
dfs = []
for subject in subjects:
    sum = pd.read_csv('Rawsummary' + str(subject) + '.csv')
    dfs.append(sum)
raw_df = pd.concat(dfs)
print('total', raw_df.isnull().mean_offset.sum())
ys = []
for subject in subjects:  # print unsuccessful trials
    subject_trials = raw_df[raw_df.subject_num == subject]
    fail_count = subject_trials.isnull().mean_offset.sum()
    ys.append(fail_count)
    print('fail count p', subject, '->', fail_count)

fig = go.Figure([go.Bar(x=list(subjects), y=ys)])
fig.update_layout(title_text='failure count by subject')
fig.show()

rep_small = [0, 2, 4, 6, 8]
rep_large = [1, 3, 5, 7, 9]
ys = []
for rep in rep_small:
    raw_df[raw_df.repetition == rep].isnull().mean_offset.sum()
    fail_count = raw_df[raw_df.repetition == rep].isnull().mean_offset.sum()
    ys.append(fail_count)
    print('SMALL fail count rep', rep, '->', fail_count)
fig = go.Figure([go.Bar(x=list(rep_small), y=ys)])
fig.update_layout(title_text='failure count by repetition-small')
fig.show()
ys = []
for rep in rep_large:
    raw_df[raw_df.repetition == rep].isnull().mean_offset.sum()
    fail_count = raw_df[raw_df.repetition == rep].isnull().mean_offset.sum()
    print('LARGE fail count rep', rep, '->', fail_count)
    ys.append(fail_count)
fig = go.Figure([go.Bar(x=list(rep_small), y=ys)])
fig.update_layout(title_text='failure count by repetition-large')
fig.show()
by_repetition = raw_df.groupby([raw_df.repetition, raw_df.cursor_type]).mean()
# print(raw_df.isnull().mean_offset.sum())
# %%


# %% basic summary plot

# subjects=[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17]
subjects = range(24)

dfs = []
for subject in subjects:
    summary = pd.read_csv("summary" + str(subject) + ".csv")
    dfs.append(summary)
    # fig = px.bar(summary, x='cursor_type', y=['mean_offset', 'std_offset', 'initial_contact_time'], barmode='group',
    #              facet_row='posture', facet_col='wide', title='summary '+str(subject))
    #
    # fig.show()
df_average = pd.concat(dfs)
fs = df_average.groupby([df_average['posture'], df_average['cursor_type'], df_average['wide']]).mean()
fs.to_csv("total_summary.csv")
fs = pd.read_csv('total_summary.csv')

fig = px.bar(fs, x='cursor_type', y=['mean_offset', 'std_offset', 'initial_contact_time'], barmode='group',
             facet_row='posture', facet_col='wide', title='total summary')

fig.show()
fs = df_average.groupby([df_average['posture'], df_average['cursor_type']]).mean()
fs.to_csv('total_summary_overall.csv')
fs = pd.read_csv('total_summary_overall.csv')
fig = px.bar(fs, x='cursor_type', y=['mean_offset', 'std_offset', 'initial_contact_time'], barmode='group',
             facet_col='posture', title='overall')
fig.show()

# for col in fs.columns:
#     if col in ['posture', 'cursor_type', 'wide']:
#         pass
#     else:
columns = fs.columns
fig = px.bar(fs, x='cursor_type', y=[x for x in fs.columns], barmode='group',
             facet_col='posture', title='overall:')
fig.show()

# %%


# %%
