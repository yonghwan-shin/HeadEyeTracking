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
d = read_hololens_data(subject=0, posture='STAND', cursor_type='NEWSPEEDSTICKY', repetition=0, secondstudy=True,
                       targetType='GRID', reset=False)
s = split_target(d, True)

# data = read_hololens_data(sub_num, pos, ct, rep, secondstudy=True, targetType=tt)
# %%
data = s[5]
data.reset_index(inplace=True)
data.timestamp -= data.timestamp.values[0]
data[['score_0', 'score_1', 'score_2', 'score_3'
    , 'score_4', 'score_5', 'score_6', 'score_7']] = pd.DataFrame(data.scores.tolist(),
                                                                  index=data.index)
# plt.plot(data.timestamp,data.score_5);plt.show()
score_columns = ['score' + str(tn) for tn in range(8)]
scores = pd.DataFrame(data.scores.to_list(), columns=score_columns,
                      index=data.index)
data = pd.concat([data, scores], axis=1)
data['selected_target'] = data.scores.apply(np.argmax)
data['stick_success'] = data['selected_target'] == 5
for k, g in itertools.groupby(data.iterrows(),
                              key=lambda row: row[1]['stick_success']):
    if k == True:
        df = pd.DataFrame([r[1] for r in g])
        # success_dwells.append(df)
for c in ['score_0', 'score_1', 'score_2', 'score_3'
    , 'score_4', 'score_5', 'score_6', 'score_7']:
    if c=='score_5':
        plt.plot(data.timestamp,data[c],'r-')
    else:
        plt.plot(data.timestamp, data[c], 'k-')
plt.show()
plt.plot(data.timestamp,data.selected_target);plt.show()
# %%
# rep_small = [0, 2, 4, 6, 8]
subjects = [1]
summary,final_summary = summarize_second_study(sub_num=0, cursorTypes=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'],
                                 postures=['STAND'], targetTypes=['GRID'], repetitions=[0]
                                 )
# %%
summary = pd.read_csv('second_Rawsummary1.csv')

fs = summary.groupby([summary.cursor_type]).mean()
fs.reindex(['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'])
import seaborn as sns

# sns.histplot(data=summary, x="longest_dwell_time", hue="cursor_type")
sns.boxplot(data=summary, x='cursor_type', y='longest_dwell_time')
plt.show()
# for ct in ['HEAD','NEWSPEED','NEWSTICKY','NEWSPEEDSTICKY']:
#     plt.hist(summary[summary.cursor_type==ct].longest_dwell_time,bins=50,label=ct)
# plt.legend()
# plt.show()
# %%
data = read_hololens_data(subject=1, posture='WALK', cursor_type='NEWSTICKY', repetition=0, secondstudy=True)
splited_data = split_target(data)
temp = splited_data[1]
# temp.reset_index(drop=True)
score_columns = ['score' + str(t) for t in range(9)]
tt = pd.DataFrame(temp.scores.to_list(), columns=score_columns, index=temp.index)
# tt.reset_index(drop=True)
temp = pd.concat([temp, tt], axis=1)
temp['selected_target'] = temp.scores.apply(np.argmax)
temp['stick_success'] = temp['selected_target'] == 1
# temp['stick_success'].plot();plt.show()
