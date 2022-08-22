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
import seaborn as sns

# from scipy.spatial.transform import Rotation as R

pio.renderers.default = 'browser'

pd.set_option('mode.chained_assignment', None)

# %%
# subjects = [15]
subjects = range(16)
for s in subjects:
    summary, final_summary = summarize_second_study(sub_num=s,
                                                    # cursorTypes=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'],
                                                    # targetTypes=['GRID', 'MENU'],
                                                    repetitions=[2, 3, 4]
                                                    )
# %%
# 5?6?7?10?
dfs = []
# ss=[15]
ss = range(16)
for subject in ss:
    summary = pd.read_csv('second_Rawsummary' + str(subject) + '.csv')
    dfs.append(summary)
    fs = summary.groupby([summary.target_type, summary.cursor_type]).mean()

    # fs= fs.reset_index()
    print(subject, '\n', fs.success_trial * 100)
summary = pd.concat(dfs)
errors = summary[summary.error.isna() == False]
summary = summary[summary.error.isna() == True]

summary = summary[summary.repetition > 1]
fs = summary.groupby([summary.cursor_type, summary.target_type]).mean()

print('Total', '\n', fs.success_trial * 100)
print(errors.error.groupby([errors.cursor_type, errors.target_type]).count())
sns.catplot(kind='bar', data=fs.success_trial.reset_index(), col='target_type', x='cursor_type', y='success_trial');
plt.show()
# print('\n',fs.success_trial*100)
# for n in ['longest_dwell_time','drop_count','initial_contact_time','success_trial','mean_dwell_time']:
#     sns.boxplot(data=summary,x='target_type',hue='cursor_type',y=n,
#                 showfliers=True,
#                             showmeans=True,
#                             meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
#                                        'markersize': '10'}
#                 )
#     plt.title(n)
#     plt.show()

# %%
subjects = [0, 1, 2, 3, 4, 5, 6]
dfs = []
for subject in subjects:
    summary_subject = pd.read_csv('second_Rawsummary' + str(subject) + '.csv')
    dfs.append(summary_subject)
summary = pd.concat(dfs)
# summary=summary.append(summary2)
fs = summary.groupby([summary.target_type, summary.cursor_type]).mean()
# fs.reindex(['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'])

# sns.boxplot(data=summary, x='cursor_type', y='longest_dwell_time', showfliers=False,
#             showmeans=True,
#             meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
#                        'markersize': '10'}, hue='target_type')
# plt.show()
for c in ['longest_dwell_time',
          'initial_contact_time', 'mean_offset',
          # 'std_offset', 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal',
          # 'std_offset_vertical',
          'trial_time', 'drop_count',
          'drop_out_count', 'mean_out_time'
          # 'abs_mean_offset_horizontal', 'abs_mean_offset_vertical'
          ]:
    sns.boxplot(data=summary, x='target_type', y=c, showfliers=False,
                showmeans=True,
                meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '10'}, hue='cursor_type')
    plt.show()

# %% corr plot

for tt in ['GRID', 'MENU', 'PIE']:
    df = summary[summary.target_type == tt].drop(['Unnamed: 0', 'posture', 'target_num', 'error', 'success_trial'],
                                                 axis=1).corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    # mask = np.zeros_like(df, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    sns.clustermap(df,
                   annot=True,
                   # mask=mask,
                   linewidths=0.5,
                   cmap='Greens',
                   vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()

# %% drop position plot
from ast import literal_eval

for tt in ['GRID', 'MENU', 'PIE']:
    for i in range(9):
        df = summary[(summary.target_type == tt) & (summary.target_num == i)]
        xs = []
        ys = []
        for x in df.drop_positions:
            try:
                x = literal_eval(x)
                for c in x:
                    # print(c)
                    xs.append(float(c[0]))
                    ys.append(float(c[1]))
            except Exception as e:
                pass
                # print(e.args)
        plt.scatter(xs, ys)
        plt.show()
    # plt.scatter()
