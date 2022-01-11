# %% IMPORTS
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from collections import defaultdict

import scipy.stats

from FileHandling import *

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from AnalysisFunctions import *

# from scipy.spatial.transform import Rotation as R

pio.renderers.default = 'browser'

pd.set_option('mode.chained_assignment', None)  # <==== 경고를 끈다

# %%

data = read_hololens_data(11, 'STAND', 'EYE', 7)
# data['cursor_speed'] = data.cursor_angular_distance.diff(1) / data.timestamp.diff(1)
# data['cursor_speed'] = abs(data.cursor_speed.rolling(10,center=True).mean())
splited_data = split_target(data)

# for t in range(3):
for t in [6]:
    temp_data = splited_data[t]
    temp_data.reset_index(inplace=True)
    temp_data.timestamp -= temp_data.timestamp.values[0]
    drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
            temp_data['direction_z'] == 0)].index
    # if len(drop_index) > 0:
    #     loss_indices = set(list(drop_index) + list(drop_index + 1) + list(drop_index + 2))
    #     if len(temp_data) in loss_indices:
    #         loss_indices.remove(len(temp_data))
    #     if len(temp_data) + 1 in loss_indices:
    #         loss_indices.remove(len(temp_data) + 1)
    #     temp_data.loc[loss_indices] = np.nan
    #     temp_data = temp_data.interpolate()

    only_success = temp_data[temp_data.cursor_angular_distance < default_target_size]
    if len(only_success) <= 0:
        continue
        # raise ValueError('no success frames', len(only_success))

    initial_contact_time = only_success.timestamp.values[0]
    dwell_time = 0.8
    frame = int(dwell_time * 60)
    maxes = []
    for i in range(len(temp_data.cursor_angular_distance) - frame):
        maxes.append(max(temp_data.cursor_angular_distance[i:i + frame]))
    print(min(maxes))
    plt.plot(temp_data.timestamp,temp_data.cursor_angular_distance)
    plt.plot(temp_data.timestamp[frame:],maxes)
    plt.axhline(min(maxes))
    plt.show()
    # maxes = [m for m in temp_data['cursor_angular_distance']]
    # targeting = temp_data[temp_data.timestamp <= initial_contact_time]
    # # temp_data.horizontal_offset.plot()
    # # temp_data.vertical_offset.plot()
    # # movement length
    # movement = (targeting.horizontal_offset.diff(1) ** 2 + targeting.vertical_offset.diff(1) ** 2).apply(math.sqrt)
    # # print(movement.sum(),t)
    # # contact position
    # contact_frame = temp_data[temp_data.timestamp == initial_contact_time]
    # x = contact_frame.horizontal_offset.values[0]
    # y = contact_frame.vertical_offset.values[0]
    # entering_position = (x, y)

    # fig, ax = plt.subplots()
    # x_offset = 1.5 * math.sin(t * math.pi / 9 * 2)
    # y_offset = 1.5 * math.cos(t * math.pi / 9 * 2)
    # # ax.scatter(x_offset,y_offset,marker='x')
    # ax.scatter(x,y,marker='o')
    # circle2 = plt.Circle((0, 0), 1.5, color='b', fill=False)
    # ax.add_patch(circle2)
    # plt.title(str(t))
    # plt.show()

# %%
offsets = visualize_offsets(False)
total_error = 0
total_len = 0
for ct, pos in itertools.product(['EYE', 'HAND', 'HEAD'], ['STAND', 'WALK']):

    h = offsets[(offsets.posture == pos) & (offsets.cursor_type == ct)]['horizontal']
    hlist = []
    for i in h.values:
        hlist += list(i)
    v = offsets[(offsets.posture == pos) & (offsets.cursor_type == ct)]['vertical']
    vlist = []
    for i in v.values:
        vlist += list(i)
    kwargs = dict(alpha=0.5, bins=100)
    # plt.hist(hlist,**kwargs,color='g',label='horizontal')
    # plt.hist(vlist, **kwargs,color='r',label='vertical')
    # plt.legend()
    # plt.show()
    # print(ct,pos)
    # herrors = [i for i in hlist if (abs(i) > 3 * sigmas[(ct, pos, 'horizontal')])]
    # verrors = [i for i in vlist if (abs(i) > 3 * sigmas[(ct, pos, 'vertical')])]
    # # verror  = [i in i in vlist if (abs(i) > 3* sigmas[(ct,pos,'vertical')])]
    # print(len(hlist),100*len(herrors)/len(hlist),100*len(verrors)/len(hlist))
    error_count = 0
    hmean = np.mean(hlist)
    vmean = np.mean(vlist)
    for i in range(len(hlist)):
        if (abs(hlist[i] - hmean) > 3 * sigmas[(ct, pos, 'horizontal')]) or (
                abs(vlist[i] - vmean) > 3 * sigmas[(ct, pos, 'vertical')]):
            error_count += 1
    print(ct, pos)
    print(len(hlist), error_count)
    total_len += len(hlist)
    total_error += error_count
    # errors = [i for i in o if (i > 3 * sigmas[(ct, pos, hv)] or i < -1 * 3 * sigmas[(ct, pos, hv)])]
    # print(ct, pos, hv)
    # print(len(o), len(errors))
print(total_len, total_error, 100 * total_error / total_len)
# %%
o = []
for i in t.values:
    o += list(i)
    # print(i)
    # print(type(i))
# collect_offsets()
# %%
for i in range(24):
    summarize_subject(i)
# %%x
summary = visualize_summary(show_plot=False)
# %%basic performance results : maybe table?
fs = summary.groupby([summary.posture, summary.cursor_type, summary.wide]).mean()
fs.to_csv('table_candidate_wide.csv')
fs_stds = summary.groupby([summary.posture, summary.cursor_type, summary.wide]).std()
fs_stds.to_csv('table_candidate_wide_stds.csv')
fs_no_wide = summary.groupby([summary.posture, summary.cursor_type]).mean()
fs_no_wide_std = summary.groupby([summary.posture, summary.cursor_type]).std()
# %% box plots?
import seaborn as sns

cs = ['mean_offset', 'std_offset',
      'initial_contact_time',
      # 'mean_offset_horizontal', 'mean_offset_vertical',
      # 'std_offset_horizontal', 'std_offset_vertical'
      ]


# from scipy import stats
# s = summary[summary.error.isna()]
# for c in cs:
#     k2,p=stats.normaltest(s[c])
#     alpha=1e-3
#     print(c)
#     if p<alpha:
#         print(p,'can be rejected')
#     else:
#         print(p,'cannot be rejected')
def replace(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = np.nan
    return group


for c in cs:
    # sns.boxplot(x='cursor_type',y=c,data=summary)
    # summary[c] = replace(summary[c], 2)
    sns.catplot(x='cursor_type', y=c, hue='posture', data=summary, kind='box', showfliers=True, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '10'})
    # sns.displot(data=summary,x=c,hue='cursor_type',col='posture',kind='kde')
    # sns.displot(data=summary,y=c,hue='posture',col='cursor_type',kind='kde')
    plt.show()

# %% RM anova
from statsmodels.stats.anova import AnovaRM
from scipy.stats import shapiro, levene, bartlett
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd

s = summary[summary.error.isna()]
# print(levene(
#     s[(s.posture == 'STAND') & (s.cursor_type == 'EYE') & (s.wide == 'LARGE')].mean_offset.values,
#     s[(s.posture == 'STAND') & (s.cursor_type == 'EYE') & (s.wide == 'SMALL')].mean_offset.values,
#     # s[(s.posture == 'STAND') & (s.cursor_type == 'EYE') & (s.wide == 'LARGE')].mean_offset.values,
# ))

cs = ['mean_offset', 'std_offset',
      'initial_contact_time',
      'target_in_count', 'target_in_total_time', 'target_in_mean_time',
      'mean_offset_horizontal', 'mean_offset_vertical',
      'std_offset_horizontal', 'std_offset_vertical', 'longest_dwell_time', 'movement_length']
for c in cs:
    # equal variance test
    print(bartlett(
        s[(s.posture == 'STAND') & (s.cursor_type == 'HEAD') & (s.wide == 'LARGE')][c].values,
        s[(s.posture == 'STAND') & (s.cursor_type == 'EYE') & (s.wide == 'LARGE')][c].values,
        s[(s.posture == 'STAND') & (s.cursor_type == 'HAND') & (s.wide == 'LARGE')][c].values,
    ))
    print(levene(
        s[(s.posture == 'STAND') & (s.cursor_type == 'HEAD') & (s.wide == 'LARGE')][c].values,
        s[(s.posture == 'STAND') & (s.cursor_type == 'EYE') & (s.wide == 'LARGE')][c].values,
        s[(s.posture == 'STAND') & (s.cursor_type == 'HAND') & (s.wide == 'LARGE')][c].values,
    ))
    wide_result = AnovaRM(data=s, depvar=c, subject='subject_num',
                          within=['wide', 'posture', 'cursor_type'], aggregate_func='mean').fit()
    print(c, wide_result)
# summary_dataframe = visualize_offsets(show_plot=False)
# target_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# %%dwell-wise anaylsis
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dwell_times=[0.1]
for dt in dwell_times:
    dwell_time_analysis(dt)
# %% visualize dwell-wise analysis
# 527          527         0.1  ...        170.893736   NaN
# 3732        3732         0.1  ...        189.193031   NaN
# 4088        4088         0.1  ...        146.179401   NaN
# 5064        5064         0.1  ...        178.581023   NaN
# 7654
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dfs = []
for dt in dwell_times:
    f = pd.read_csv('dwell_time_Rawsummary' + str(dt) + '.csv')
    f['dwell_time'] = dt
    dfs.append(f)
dwell_summary = pd.concat(dfs)
plot_df = pd.DataFrame(
    columns=['posture', 'cursor_type', 'dwell_time', 'success_rate', 'required_target_size', 'first_dwell_time',
             'mean_final_speed', 'required_target_size_std', 'first_dwell_time_std', 'mean_final_speed_std'])
for pos, ct in itertools.product(['STAND', 'WALK'], ['HEAD', 'EYE', 'HAND']):

    srs = []
    for dt in dwell_times:
        temp = dwell_summary[(dwell_summary.dwell_time == dt) & (dwell_summary.cursor_type == ct)]
        temp = temp[temp.posture == pos]
        total_count = len(temp)
        all_error_count = sum(temp.groupby(temp.error).count().posture)

        fail_count = temp.groupby(temp.error).count().posture[0]
        # print(dt,temp.groupby(temp.error).count().posture)
        success_rate = 1 - fail_count / (total_count - (all_error_count - fail_count))
        # print(dt,pos,ct, success_rate,total_count,fail_count,all_error_count)
        # print(temp.groupby(temp.error).count())
        required_target_size = temp.required_target_size.mean()
        first_dwell_time = temp.first_dwell_time.mean()
        mean_final_speed = temp.mean_final_speed.mean()
        # errorbar
        required_target_size_std = temp.required_target_size.std()
        first_dwell_time_std = temp.first_dwell_time.std()
        mean_final_speed_std = temp.mean_final_speed.std()
        plot_summary = {'posture': pos,
                        'cursor_type': ct,
                        'dwell_time': dt,
                        'success_rate': success_rate * 100,
                        'required_target_size': required_target_size,
                        'first_dwell_time': first_dwell_time,
                        'mean_final_speed': mean_final_speed,
                        'required_target_size_std': required_target_size_std,
                        'first_dwell_time_std': first_dwell_time_std,
                        'mean_final_speed_std': mean_final_speed_std
                        }
        plot_df.loc[len(plot_df)] = plot_summary
dwell_summary.to_csv('DwellRawSummary.csv')
# %%

for c in ['success_rate', 'required_target_size', 'first_dwell_time',
          'mean_final_speed']:
    fig = go.Figure()
    for pos, ct in itertools.product(['STAND', 'WALK'], ['HEAD', 'EYE', 'HAND']):
        plot_data = plot_df[(plot_df.posture == pos) & (plot_df.cursor_type == ct)]
        # print(plot_data)
        erry = plot_data[c + '_std'] if c != 'success_rate' else None
        fig.add_trace(go.Bar(
            name=pos + '_' + ct, x=dwell_times, y=plot_data[c], error_y=dict(type='data', array=erry)
        ))
        fig.update_layout(title=str(c))
    fig.show()
# %%dwell based plots
# success rate plot
import seaborn as sns

sns.set_style('ticks')
sns.set_context('notebook')
fig, ax = plt.subplots()
sns.lineplot(x='dwell_time', y='success_rate', hue='cursor_type', style='posture', data=plot_df, marker='o',
             markersize=10, alpha=0.75,
             palette='Set2', linewidth=3)
plt.ylim(0, 110)

plt.xticks(dwell_times)
plt.xlabel("Dwell Threshold (s)")
plt.ylabel('Success Rate (%)')
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, ['Cursor Type', 'Head', 'Eye', 'Hand', 'Posture', 'STAND', 'WALK'], loc='center right')

plt.show()
# %% dwell time box plots
dcs = ['required_target_size', 'first_dwell_time',
       'mean_final_speed','min_target_size']

sns.set_style('ticks')
sns.set_context('talk')
# sns.catplot(data=plot_data,x='dwell_time',y='success_rate',hue='')
for c in dcs:
    # Walking condition plots
    sns.boxplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary[dwell_summary.posture == 'WALK'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '10'},
                palette='Set1', width=0.8)
    plt.legend(loc='upper right')
    if c == 'required_target_size':
        plt.ylabel('Required Target Size (°)')
        # plt.ylim(0, 13)
    elif c == 'first_dwell_time':
        plt.ylabel('First Dwell Success Time (s)')
        # plt.ylim(0, 3.5)
    elif c == 'mean_final_speed':
        plt.ylabel('Final Cursor Speed (°/s)')
        # plt.ylim(0, 5)
        # plt.title('Walk '+ c)
    # xmin, xmax, ymin, ymax = plt.axis()
    # plt.ylim(ymin - 1, ymax)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
               fancybox=False, ncol=3)

    plt.xlabel('dwell threshold (s)')
    plt.show()
    # Standing condition plots
    sns.boxplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary[dwell_summary.posture == 'STAND'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '10'},
                palette='Set2', width=0.8)
    # sns.catplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary[dwell_summary.posture=='WALK'], kind='box', showfliers=False, showmeans=True,
    #             meanprops={'marker': '+', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '5'},
    #             height=5,aspect=2,legend=False)
    # sns.catplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary[dwell_summary.posture=='STAND'], kind='box', showfliers=False, showmeans=True,
    #             meanprops={'marker': '+', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '5'},
    #             height=5,aspect=2,legend=False)
    # sns.displot(data=summary,x=c,hue='cursor_type',col='posture',kind='kde')
    # sns.displot(data=summary,y=c,hue='posture',col='cursor_type',kind='kde')
    if c == 'required_target_size':
        plt.ylabel('Required Target Size (°)')
        # plt.ylim(0, 11)
    elif c == 'first_dwell_time':
        plt.ylabel('First Dwell Success Time (s)')
        # plt.ylim(0, 5)
    elif c == 'mean_final_speed':
        plt.ylabel('Final Cursor Speed (°/s)')
        plt.ylim(0, 20)
    # plt.legend(loc='upper right')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
               fancybox=True, ncol=3)
    plt.xlabel('dwell threshold (s)')
    plt.show()

# %%
fig = px.bar(plot_df, x='dwell_time', y=['success_rate', 'required_target_size', 'first_dwell_time',
                                         'mean_final_speed'], barmode='group', facet_col='posture',
             facet_row='cursor_type',
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

i = 2
print(i)
without_stand = summarize_subject(i, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], True, False)
with_stand = summarize_subject(i, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], False, False)
without_walk = summarize_subject(i, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], True, False)
with_walk = summarize_subject(i, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], False, False)
print('stand,accuracy', with_stand.mean_offset.mean(), '->', without_stand.mean_offset.mean())
print('walk,accuracy', with_walk.mean_offset.mean(), '->', without_walk.mean_offset.mean())
print('stand,precision', with_stand.std_offset.mean(), '->', without_stand.std_offset.mean())
print('walk,precision', with_walk.std_offset.mean(), '->', without_walk.std_offset.mean())

i = 4
j = 6

print(i)
without_stand = summarize_subject(i, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], False, False)
with_stand = summarize_subject(j, ['EYE'], ['STAND'], range(9), [4, 5, 6, 7, 8, 9], False, False)
without_walk = summarize_subject(i, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], False, False)
with_walk = summarize_subject(j, ['EYE'], ['WALK'], range(9), [4, 5, 6, 7, 8, 9], False, False)
print('stand,accuracy', with_stand.mean_offset.mean(), '->', without_stand.mean_offset.mean())
print('walk,accuracy', with_walk.mean_offset.mean(), '->', without_walk.mean_offset.mean())
print('stand,precision', with_stand.std_offset.mean(), '->', without_stand.std_offset.mean())
print('walk,precision', with_walk.std_offset.mean(), '->', without_walk.std_offset.mean())
