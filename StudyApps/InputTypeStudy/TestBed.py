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

pd.set_option('mode.chained_assignment', None)

# %%
# dd=summarize_subject(2, resetFile=False, suffix='Triangle' + str(5), fnc=TriangleDataframe, arg=5)
for t in np.arange(5, 65, 5):
    for i in range(24):
        summarize_subject(i, resetFile=False, suffix='Moving' + str(t), fnc=MovingAverage, arg=t)
# %%
dfs = []
data = visualize_summary(show_plot=False, subjects=range(24))
# data=visualize_summary(show_plot=False,subjects=[2])
data['window'] = 0
data = data[data.cursor_type != 'NEW']
dfs.append(data)
for t in np.arange(5, 65, 5):
    data = visualize_summary(show_plot=False, subjects=range(24), suffix='Moving' + str(t))
    # data = visualize_summary(show_plot=False, subjects=[2], suffix='Triangle' + str(t))
    data['window'] = t
    dfs.append(data)
summary = pd.concat(dfs)
fs = summary.groupby([summary.window, summary.posture, summary.cursor_type]).mean()
fs = fs.reset_index()
parameters = list(fs.columns)
remove_columns = ['Unnamed: 0', 'subject_num', 'repetition', 'target_num']
for removal in remove_columns:
    parameters.remove(removal)
# %%

# parameters=['mean_offset']
# for p in parameters:
fig = px.bar(fs, x='window', y=parameters, barmode='group', facet_col='posture', facet_row='cursor_type')
fig.show()

# fig = px.bar(plot_df, x='dwell_time', y=['success_rate', 'required_target_size', 'first_dwell_time',
#                                          'mean_final_speed'], barmode='group', facet_col='posture',
#              facet_row='cursor_type',
#              title='target_size')
# %%
# 20 WALK EYE 6 6
# 14 WALK HEAD 4 2
# 15 WALK HAND 6 3
# 17 WALK EYE 4 2
# 0 WALK EYE 8 8
# temp_data = get_one_trial(20, 'WALK', 'EYE', 6, 6)
# temp_data = get_one_trial(14 ,'WALK', 'HEAD', 4, 2)
# temp_data = get_one_trial(15 ,'WALK', 'HAND', 6, 3)
# temp_data = get_one_trial(17 ,'WALK', 'EYE', 4, 2)
# temp_data=check_loss(temp_data,'EYE')
temp_data = get_one_trial(0, 'WALK', 'HEAD', 8, 7)

# temp_data = read_hololens_data(22, 'WALK', 'EYE', 5, reset=True)

# temp_data= get_one_trial(22,'WALK','HEAD',5,8)
plt.plot(temp_data.timestamp,temp_data.head_forward_y)
plt.plot(temp_data.timestamp,temp_data.direction_y)
plt.show()
plt.plot(temp_data.timestamp, temp_data.horizontal_offset, 'b-', label='H')
plt.plot(temp_data.timestamp, temp_data.vertical_offset, 'r-', label='V')
plt.plot(temp_data.timestamp, temp_data.cursor_angular_distance, 'c-', label='A', alpha=0.5)
plt.plot(temp_data.timestamp, temp_data.max_angle, 'k--')
plt.plot(temp_data.timestamp, -temp_data.max_angle, 'k--')
# plt.plot(temp_data.timestamp,temp_data.target_horizontal_velocity)
# plt.ylim(-15,15)
plt.axhline(0)
plt.axhline(temp_data.cursor_angular_distance.mean())
plt.legend()
plt.show()


def c(a):
    if a < -90:
        return a + 360
    else:
        return a


# plt.plot(temp_data.timestamp,temp_data.horizontal_offset,'b-',label='H')
# plt.plot(temp_data.timestamp, temp_data.cursor_horizontal_angle.apply(c), 'r-', label='V')
# plt.plot(temp_data.timestamp, temp_data.target_horizontal_angle, 'b-', label='T')
# plt.legend()
# plt.show()

# %%
# data = read_hololens_data(11, 'STAND', 'EYE', 7)
# temp_data = get_one_trial(20,'WALK','EYE',5,2)
summary = pd.read_csv("BasicRawSummary.csv")
jumps = summary[summary.error == 'jump']
for trial in jumps.head(3).iterrows():
    trial = trial[1]
    temp_data = get_one_trial(trial.subject_num, trial.posture, trial.cursor_type, trial.repetition, trial.target_num)
    outlier = list(temp_data[(abs(temp_data.target_horizontal_velocity) > 10 * 57.296)].index)
    outlier = [x for x in outlier if x > 5]
    outlier_timestamp = temp_data.iloc[outlier].timestamp.values
    plt.plot(temp_data.timestamp, temp_data.angle, 'r:')
    plt.plot(temp_data.timestamp, temp_data.max_angle, 'b:')
    corr_data = check_loss(temp_data, trial.cursor_type)
    corr_data['target_horizontal_velocity'] = (
            corr_data['target_horizontal_angle'].diff(1).apply(correct_angle) / corr_data['timestamp'].diff(1))
    outlier = list(corr_data[(abs(corr_data.target_horizontal_velocity) > 10 * 57.296)].index)
    outlier = [x for x in outlier if x > 5]
    corr_outlier_timestamp = corr_data.iloc[outlier].timestamp.values
    plt.plot(corr_data.timestamp, corr_data.angle, 'r')
    plt.plot(corr_data.timestamp, corr_data.max_angle, 'b')
    plt.title(str(corr_outlier_timestamp))
    # for ol in outlier_timestamp:
    #     plt.axvline(ol)
    for ol in corr_outlier_timestamp:
        plt.axvline(ol)
    plt.show()

    # plt.plot(corr_data.timestamp,abs(corr_data.target_horizontal_velocity))
    plt.plot(corr_data.timestamp, corr_data.target_horizontal_velocity)
    plt.show()
# %% see eye-errors
repetitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

postures = ['STAND', 'WALK']

cursorTypes = ['EYE']
targets = range(9)
for sub_num in range(24):
    # for sub_num in [6]:
    subject_total_count = 0
    subject_error_eye_count = 0
    subject_error_invalidate = 0
    for cursor_type, rep, pos in itertools.product(cursorTypes, repetitions, postures):
        data = read_hololens_data(sub_num, pos, cursor_type, rep)
        splited_data = split_target(data)
        # wide = 'SMALL' if rep in rep_small else 'LARGE'
        for t in targets:
            try:

                temp_data = splited_data[t]
                temp_data.reset_index(inplace=True)
                temp_data.timestamp -= temp_data.timestamp.values[0]
                validate, reason = validate_trial_data(temp_data, cursor_type, pos)
                temp_data['check_eye'] = temp_data.latestEyeGazeDirection_x.diff(1)
                eye_index = temp_data[temp_data.check_eye == 0].index
                invalidate_index = temp_data[temp_data.isEyeTrackingEnabledAndValid == False].index
                subject_total_count += len(temp_data.index)
                subject_error_eye_count += len(eye_index)
                subject_error_invalidate += len(invalidate_index)
                # plt.plot(temp_data.timestamp,temp_data.horizontal_offset)
                # plt.plot(temp_data.timestamp, temp_data.vertical_offset)
                # plt.show()
                # plt.plot(temp_data.timestamp,temp_data.direction_x)
                # plt.plot(temp_data.timestamp, temp_data.head_forward_x)
                # plt.show()
                # if len(eye_index)>0:
                #     print(sub_num, pos, cursor_type, rep,len(eye_index))
            except Exception as e:
                print(sub_num, pos, cursor_type, rep, e.args)
    print(sub_num, subject_error_eye_count, subject_total_count, subject_error_invalidate,
          100 * subject_error_eye_count / subject_total_count, '%',
          100 * subject_error_invalidate / subject_total_count, '%')

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
# for t in np.arange(5, 65, 5):
for i in [1, 3]:
    summarize_subject(i, resetFile=False)
# summary = summarize_subject(6)

# summary = visualize_summary(show_plot=False)
# summary.to_csv('BasicRawSummary.csv')
# errors = summary[summary.error.isna() == False]
# print('\nError trial counts')
# print(errors.groupby(errors['error']).subject_num.count())
# %%
import seaborn as sns

summary = pd.read_csv('BasicRawSummary.csv')
summary = summary[summary.error.isna()]
# summary=summary[(summary.posture=='WALK')]
# for trial in summary.sort_values(by=['std_offset'],ascending=False).head(3).iterrows():
#     trial = trial[1]
#     temp_data = get_one_trial(trial.subject_num, trial.posture, trial.cursor_type, trial.repetition, trial.target_num)
#     plt.plot(temp_data.timestamp, temp_data.horizontal_offset)
#     plt.plot(temp_data.timestamp, temp_data.vertical_offset)
#     plt.plot(temp_data.timestamp, temp_data.cursor_angular_distance,'+')
#     plt.title(str(trial.subject_num) +  trial.posture +  trial.cursor_type + str(trial.repetition) + str(trial.target_num))
#     plt.show()
#     plt.plot(temp_data.timestamp, temp_data.target_horizontal_angle)
#     plt.plot(temp_data.timestamp, temp_data.cursor_horizontal_angle)
#     plt.plot(temp_data.timestamp, temp_data.target_vertical_angle, '.')
#     plt.plot(temp_data.timestamp, temp_data.cursor_vertical_angle, '.')
#     plt.show()
sns.boxplot(data=summary, hue='cursor_type',
            x='posture',
            y='std_offset',
            showfliers=False,
            showmeans=True,
            )
plt.show()
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
from scipy.stats import shapiro
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
from scipy.stats import anderson
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from bioinfokit.analys import stat
# import pyvttbl as pt
from collections import namedtuple
from statannotations.Annotator import Annotator

pd.set_option('max_colwidth', 400)
pd.set_option('max_rows', 99999)
summary = pd.read_csv("BasicRawSummary.csv")
summary['accuracy'] = (summary.mean_offset_horizontal ** 2 + summary.mean_offset_vertical ** 2).apply(math.sqrt)
for column in [
    # 'mean_offset',
    'std_offset',
    'initial_contact_time',
    'target_in_count',
    # 'target_in_total_time',
    # 'target_in_mean_time',
    'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal',
    'std_offset_vertical',
    # 'mean_abs_offset_horizontal', 'mean_abs_offset_vertical',
    # 'std_abs_offset_horizontal', 'std_abs_offset_vertical',
    'longest_dwell_time', 'movement_length', 'accuracy'
]:

    s = summary[summary.error.isna()]
    # ss = s

    list_subjects = []
    for i, subject_data in s.groupby(s.subject_num):
        d = subject_data[np.abs(stats.zscore(subject_data[column])) < 3]
        list_subjects.append(d.reset_index())
    s = pd.concat(list_subjects)

    bySubjects = s.groupby([s.subject_num, s.cursor_type, s.posture]).mean()
    bySubjects = bySubjects.reset_index()
    if column == 'initial_contact_time':
        aovrm = AnovaRM(s, column, 'subject_num', within=['cursor_type', 'posture', 'wide'],
                        aggregate_func='mean').fit()
    else:
        aovrm = AnovaRM(s, column, 'subject_num', within=['cursor_type', 'posture'],
                        aggregate_func='mean').fit()

    aov = pg.rm_anova(dv=column, within=['cursor_type', 'posture'],
                      subject='subject_num', data=s, detailed=True,
                      effsize="ng2", correction=True)
    sph = pg.sphericity(dv=column, within=['cursor_type', 'posture'],
                        subject='subject_num', data=s,
                        )
    pg_posthoc = pg.pairwise_ttests(dv=column, within=['posture', 'cursor_type'], subject='subject_num', data=s,
                                    padjust='bonf')

    print('METRIC', column)
    # plt.subplot(1, 2, 1)
    # sns.pointplot(data=s[s.posture == 'STAND'], x='cursor_type', y=column, hue='wide', dodge=True,
    #               linestyles=["-", "--"], order=["HAND", 'HEAD', 'EYE'])
    # plt.subplot(1, 2, 2)
    # sns.pointplot(data=s[s.posture == "WALK"], x='cursor_type', y=column, hue='wide', dodge=True,
    #               linestyles=["-", "--"], order=["HAND", 'HEAD', 'EYE'])
    # plt.show()
    order = ['STAND', 'WALK']
    hue_order = ["HAND", 'HEAD', 'EYE']
    # pairs = [(('STAND', 'HAND'), ('WALK', 'HAND')),
    #          (('STAND', 'HEAD'), ('WALK', 'HEAD'))]
    pairs = [
        (('STAND', 'HAND'), ('STAND', 'HEAD')),
        (('STAND', 'HAND'), ('STAND', 'EYE')),
        (('STAND', 'HEAD'), ('STAND', 'EYE')),
        (('WALK', 'HAND'), ('WALK', 'HEAD')),
        (('WALK', 'HAND'), ('WALK', 'EYE')),
        (('WALK', 'HEAD'), ('WALK', 'EYE')),
    ]
    ax = sns.boxplot(data=bySubjects, x='posture', y=column, hue='cursor_type', showfliers=False, showmeans=True,
                     meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                                'markersize': '10'},
                     palette='Set1', width=0.8)
    annot = Annotator(ax, pairs, data=bySubjects, x='posture', y=column, order=order, hue='cursor_type',
                      hue_order=hue_order)
    # annot.new_plot(ax, pairs, data=s, x='posture', y=column, order=order, hue='cursor_type', hue_order=hue_order)
    annot.configure(test='t-test_ind', comparisons_correction="Bonferroni").apply_test().annotate()
    # sns.boxplot(data=ss, x='posture', y=column, hue='cursor_type', showfliers=False, showmeans=True, )
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1), title='Cursor Type')
    plt.title(column)
    plt.tight_layout()
    plt.show()
    from statsmodels.graphics.factorplots import interaction_plot

    # fig = interaction_plot(x=s['posture'], trace=s['cursor_type'], response=s[column],
    #                        colors=['#4c061d', '#d17a22', '#b4c292'])
    # plt.show()
    aov.round(3)
    print('RM-anova(statsmodel)')
    print(aovrm)
    print('RM-anova(pg)')
    # print(aov)
    pg.print_table(aov)
    print('Sphericity(pg)')
    print(sph)
    # plt.figure(figsize=(20, 10))
    # bySubjects.boxplot(column=column,by=['cursor_type','posture'])
    # sns.boxplot(data=bySubjects, x='cursor_type', y=column, hue='posture')
    # plt.show()
    # display(bySubjects)
    bySubjects_wide = s.groupby([s.subject_num, s.cursor_type, s.posture, s.wide]).mean()
    bySubjects_wide = bySubjects_wide.reset_index()
    for pos in ['STAND', 'WALK']:
        print(pos, bySubjects_wide[bySubjects_wide.posture == pos].pairwise_tukey(dv=column, between=['wide']).round(3))
        print(pg.normality(data=bySubjects[bySubjects.posture == pos], dv=column, group='cursor_type'))
        # posthoc = pairwise_tukeyhsd(bySubjects[column], bySubjects['cursor_type'], alpha=0.05)
        # print('Posthoc(tukeyhsd)',pos)
        # print(posthoc)
        # fig = posthoc.plot_simultaneous()
        # fig.show()

    print('Posthoc(pairwise ttest,pg)')
    pg.print_table(pg_posthoc)
    # # final summary
    # for i in aovrm.anova_table[aovrm.anova_table['Pr > F'] < 0.05].index:
    #   print('rANOVA significant difference in ',i)
    # for i in aov[aov['p-GG-corr']<0.05].Source.values:
    #   print('rANOVA significant difference in (withouf wide)',i)
    # print('sphericity:',sph.spher, 'p:',sph.pval)
    # # print(pg_posthoc)
    print('-' * 100)

# %%dwell-wise anaylsis
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dwell_times = [1.0]
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
             'mean_final_speed', 'required_target_size_std', 'first_dwell_time_std', 'mean_final_speed_std',
             'best_record'])
for pos, ct in itertools.product(['WALK', 'STAND'], ['HEAD', 'EYE', 'HAND']):

    srs = []
    for dt in dwell_times:
        temp = dwell_summary[(dwell_summary.dwell_time == dt) & (dwell_summary.cursor_type == ct)]
        temp = temp[temp.posture == pos]
        total_count = len(temp)
        all_error_count = sum(temp.groupby(temp.error).count().posture)
        if temp.groupby(temp.error).count().posture.__contains__("no success dwell"):
            fail_count = temp.groupby(temp.error).count().posture['no success dwell']
        else:
            fail_count = 0
        # print(dt,temp.groupby(temp.error).count().posture)
        success_rate = 1 - fail_count / (total_count - (all_error_count - fail_count))
        print(dt, pos, ct, success_rate, total_count, fail_count, all_error_count)
        print(temp.groupby(temp.error).count().posture)
        required_target_size = temp.required_target_size.mean()
        first_dwell_time = temp.first_dwell_time.mean()
        mean_final_speed = temp.mean_final_speed.mean()
        mean_best_record = temp.best_record.mean()
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
                        'mean_final_speed_std': mean_final_speed_std,
                        'mean_best_record': mean_best_record
                        }
        plot_df.loc[len(plot_df)] = plot_summary
dwell_summary.to_csv('DwellRawSummary.csv')

abc = dwell_summary[
    (dwell_summary.posture == 'STAND') & (dwell_summary.cursor_type == 'HAND') & (dwell_summary.dwell_time == 1.0)]
ddd = dwell_summary[
    (dwell_summary.posture == 'STAND') & (dwell_summary.cursor_type == 'EYE') & (dwell_summary.dwell_time == 1.0)]
# %%

for c in ['success_rate', 'required_target_size', 'first_dwell_time',
          'mean_final_speed', 'best_record']:
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
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(x='dwell_time', y='success_rate', hue='cursor_type', data=plot_df,
             palette='Set2', marker='o', style='posture', ci=0, style_order=['STAND', 'WALK'],
             ax=ax)
# sns.lineplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary,
#
#              palette='Set2', marker='o', style='posture', ci=0,
#              # width=0.8,
#              ax=ax)
plt.ylim(0, 110)

plt.xticks(dwell_times)
plt.xlabel("Dwell Threshold (s)")
plt.ylabel('Success Rate (%)')
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, ['Cursor Type', 'Head', 'Eye', 'Hand', 'Posture', 'STAND', 'WALK'], loc='center right')
plt.tight_layout()
plt.show()
# %% dwell time box plots
dcs = ['required_target_size',
       # 'first_dwell_time',
       # 'mean_final_speed', 'min_target_size', 'best_record'
       ]

sns.set_style('ticks')
sns.set_context('talk')
# sns.catplot(data=plot_data,x='dwell_time',y='success_rate',hue='')
for c in dcs:
    # Walking condition plots
    # fig, ax = plt.subplots(figsize=(6, 8))
    # sns.lineplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary[dwell_summary.posture == 'WALK'],
    #             # showfliers=False, showmeans=True,
    #             # meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '10'},
    #             palette='Set2', markers=True, dashes=True,style='cursor_type',ci=0,
    #              # width=0.8,
    #              ax=ax)
    # plt.legend(loc='upper right')
    # if c == 'required_target_size':
    #     plt.ylabel('Required Target Size (°)')
    #     # plt.ylim(0, 13)
    # elif c == 'first_dwell_time':
    #     plt.ylabel('First Dwell Success Time (s)')
    #     # plt.ylim(0, 3.5)
    # elif c == 'mean_final_speed':
    #     plt.ylabel('Final Cursor Speed (°/s)')
    #     # plt.ylim(0, 5)
    #     # plt.title('Walk '+ c)
    # # xmin, xmax, ymin, ymax = plt.axis()
    # # plt.ylim(ymin - 1, ymax)
    # plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
    #            fancybox=False, ncol=3)
    #
    # plt.xlabel('dwell threshold (s)')
    # plt.tight_layout()
    # plt.show()
    # Standing condition plots
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary,
                 # showfliers=False, showmeans=True,
                 # meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                 #            'markersize': '10'},
                 palette='Set2', marker='o', style='posture', ci=0,
                 # width=0.8,
                 ax=ax)
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
        plt.xticks(dwell_summary.dwell_time.unique())
        plt.ylim(0, 10)
    elif c == 'first_dwell_time':
        plt.ylabel('First Dwell Success Time (s)')
        # plt.ylim(0, 5)
    elif c == 'mean_final_speed':
        plt.ylabel('Final Cursor Speed (°/s)')
        plt.ylim(0, 20)
    # plt.legend(loc='upper right')
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, ['Cursor Type', 'Head', 'Eye', 'Hand', 'Posture', 'STAND', 'WALK'],
              # loc='center right'
              loc='best'
              )

    # plt.legend(loc='lower left',
    #            bbox_to_anchor=(1, 0),
    # #            fancybox=True, ncol=3,
    #            )
    plt.xlabel('dwell threshold (s)')
    plt.tight_layout()
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
