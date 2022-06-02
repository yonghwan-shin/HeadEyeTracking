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
# d = read_hololens_data(subject=1, posture='WALK', cursor_type='NEWSTICKY', repetition=0, secondstudy=True)
rep_small = [0, 2, 4, 6, 8]
subjects = [1]
summary=summarize_subject(sub_num=1, cursorTypes=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'], postures=['WALK'],
                  secondstudy=True)
#%%
summary = pd.read_csv('second_Rawsummary1.csv')

fs = summary.groupby([summary.cursor_type]).mean()
fs.reindex(['HEAD','NEWSPEED','NEWSTICKY','NEWSPEEDSTICKY'])
import seaborn as sns
# sns.histplot(data=summary, x="longest_dwell_time", hue="cursor_type")
sns.boxplot(data=summary,x='cursor_type',y='longest_dwell_time')
plt.show()
# for ct in ['HEAD','NEWSPEED','NEWSTICKY','NEWSPEEDSTICKY']:
#     plt.hist(summary[summary.cursor_type==ct].longest_dwell_time,bins=50,label=ct)
# plt.legend()
# plt.show()
#%%
data = read_hololens_data(subject=1, posture='WALK', cursor_type='NEWSTICKY', repetition=0, secondstudy=True)
splited_data = split_target(data)
temp = splited_data[1]
# temp.reset_index(drop=True)
score_columns = ['score' + str(t) for t in range(9)]
tt = pd.DataFrame(temp.scores.to_list(),columns=score_columns,index=temp.index)
# tt.reset_index(drop=True)
temp = pd.concat([temp,tt],axis=1)
temp['selected_target']=temp.scores.apply(np.argmax)
temp['stick_success'] = temp['selected_target']==1
# temp['stick_success'].plot();plt.show()