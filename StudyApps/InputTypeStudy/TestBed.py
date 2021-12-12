# %% IMPORTS
import itertools
import math

import numpy as np

import pandas as pd
from collections import defaultdict
from FileHandling import *

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from AnalysisFunctions import *

# from scipy.spatial.transform import Rotation as R

pio.renderers.default = 'browser'

pd.set_option('mode.chained_assignment', None)  # <==== 경고를 끈다

# %%
t = 5
data = get_one_trial(11, 'WALK', 'HAND', 8, 1)
data['cursor_speed'] = data.cursor_angular_distance.diff(1) / data.timestamp.diff(1)
plt.plot(data.cursor_angular_distance)
plt.axhline(default_target_size)
plt.plot(data.cursor_speed.rolling(30,min_periods=1).mean())
plt.show()
# tt = data[data.target_name == 'Target_0']
# plt.plot(tt.cursor_angular_distance)
# plt.show()
# s = dwell_time_analysis(0.1)
# data = read_hololens_data(0, 'WALK', 'HEAD', t,True)
# for s in range(24):
#     # collect_offsets(s)
#     summarize_subject(s)

# summary = visualize_summary(show_plot=True)
# summary_dataframe = visualize_offsets(show_plot=False)
# target_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# %%dwell-wise anaylsis
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dwell_times=[0.1]
for dt in dwell_times:
    dwell_time_analysis(dt)
# %% visualize dwell-wise analysis
dfs = []
for dt in dwell_times:
    f = pd.read_csv('dwell_time_Rawsummary' + str(dt) + '.csv')
    f['dwell_time'] = dt
    dfs.append(f)
summary = pd.concat(dfs)
plot_df = pd.DataFrame(
    columns=['posture', 'cursor_type', 'dwell_time', 'success_rate', 'required_target_size', 'first_dwell_time',
             'mean_final_speed'])
for pos, ct in itertools.product(['STAND', 'WALK'], ['HEAD', 'EYE', 'HAND']):

    srs = []
    for dt in dwell_times:
        temp = summary[(summary.dwell_time == dt) & (summary.cursor_type == ct)]
        temp = temp[temp.posture == pos]
        total_count = len(temp)
        all_error_count = sum(temp.groupby(temp.error).count().posture)

        fail_count = temp.groupby(temp.error).count().posture[0]
        # print(dt,temp.groupby(temp.error).count().posture)
        success_rate = 1 - fail_count / (total_count - (all_error_count - fail_count))
        required_target_size = temp.required_target_size.mean()
        first_dwell_time = temp.first_dwell_time.mean()
        mean_final_speed = temp.mean_final_speed.mean()
        plot_summary = {'posture': pos,
                         'cursor_type': ct,
                         'dwell_time': dt,
                         'success_rate': success_rate,
                         'required_target_size': required_target_size,
                         'first_dwell_time': first_dwell_time,
                         'mean_final_speed': mean_final_speed
                         }
        plot_df.loc[len(plot_df)] = plot_summary
#%%
fig = px.bar(plot_df, x='dwell_time', y=['success_rate', 'required_target_size', 'first_dwell_time',
             'mean_final_speed'], barmode='group', facet_col='posture', facet_row='cursor_type',
             title='target_size')
fig.show()
# %%
fs = summary.groupby([summary.dwell_time, summary.posture, summary.cursor_type]).mean()
fs = fs.reset_index()
parameters = ['first_dwell_time',
              'target_in_count', 'target_in_total_time', 'target_in_mean_time', 'required_target_size']
fig = px.bar(fs, x='dwell_time', y=parameters, barmode='group', facet_col='posture', facet_row='cursor_type',
             title='target_size')
fig.show()
# walks=fs[fs.posture=='WALK']
# %% visualize target-size wise analysis
target_sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# for ts in target_sizes:
#     target_summary = target_size_analysis(ts) #TODO : add success rate!

dfs = []
for ts in target_sizes:
    f = pd.read_csv('target_size_summary' + str(ts) + '.csv')
    f['target_size'] = ts
    dfs.append(f)
summary = pd.concat(dfs)
fs = summary.groupby([summary.posture, summary.cursor_type, summary.target_size]).mean()
fs.to_csv('target_size_summary.csv')
fs = pd.read_csv('target_size_summary.csv')
parameters = ['initial_contact_time',
              'target_in_count', 'target_in_total_time', 'target_in_mean_time']
fig = px.bar(fs, x='target_size', y=parameters, barmode='group', facet_col='posture', facet_row='cursor_type',
             title='target_size')
fig.show()
# %% fail count
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
target_sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
dfs = []
for ts in target_sizes:
    f = pd.read_csv('target_size_Rawsummary' + str(ts) + '.csv')
    # f['target_size'] = ts
    # print(f.groupby(f.error).count(),ts)
    fail_count = f.groupby(f.error).count().values[0][0]
    all_count = len(f)
    print(ts, fail_count, '/', all_count, 100 * fail_count / all_count, '%')
    # errors[(errors.error != 'jump') & (errors.error !='loss')]
    dfs.append(f)
raw_summary = pd.concat(dfs)
# %%
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

dfs = []
for dt in dwell_times:
    success = raw_summary[raw_summary.longest_dwell > dt]
    ss = success.groupby([success.posture, success.cursor_type, success.target_size])

    c = ss.longest_dwell.count()
    c = c.reset_index()
    c['dwell_time'] = dt
    # df.reset_index(level=0, inplace=True)
    dfs.append(c)

    # print(ss.longest_dwell.count())
# fig = px.bar(fs, x='target_size', y=parameters, barmode='group', facet_col='posture', facet_row='cursor_type',
#              title='target_size')
fs = pd.concat(dfs)
fig = go.Figure()
for cursor_type in ['HEAD', 'HAND', 'EYE']:
    for ts in target_sizes:
        for dt in dwell_times:
            fff = fs[(fs.cursor_type == cursor_type) & (fs.posture == 'WALK') & (fs.target_size == ts) & (
                    fs.dwell_time == dt)]
            fig.add_trace(
                go.Bar(
                    x=fff.dwell_time, y=fff.longest_dwell
                )
            )
# fig = px.bar(fs,x='dwell_time',y=['longest_dwell','target_size'],barmode='group', facet_col='posture', facet_row='cursor_type')
# fs = fs.reset_index()
# fig  = px.bar(fs,)
# fig = go.Figure()
fig.show()
# %% pilot study: with/without eye cursor visual

# data = read_hololens_data(0, 'WALK', 'HEAD', t,False)
without_stand = summarize_subject(0, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], True, False)
with_stand = summarize_subject(23, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], False, False)
without_walk = summarize_subject(1, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], True, False)
with_walk = summarize_subject(23, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], False, False)

print('stand,accuracy', with_stand.mean_offset.mean(), '->', without_stand.mean_offset.mean())
print('walk,accuracy', with_walk.mean_offset.mean(), '->', without_walk.mean_offset.mean())
print('stand,precision', with_stand.std_offset.mean(), '->', without_stand.std_offset.mean())
print('walk,precision', with_walk.std_offset.mean(), '->', without_walk.std_offset.mean())
