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


def check_modality(string):
    if 'SPEED' in string:
        return True
    else:
        return False


def check_inten(string):
    if 'STICKY' in string:
        return True
    else:
        return False


# %%
# subjects = [13, 14]
subjects = range(16)
for s in subjects:
    summary, final_summary = summarize_second_study(sub_num=s,
                                                    # cursorTypes=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'],
                                                    # targetTypes=['GRID', 'MENU'],
                                                    repetitions=[0, 1, 2, 3, 4], saveFile=True, pickle=False,
                                                    )
# %%
# 5?6?7?10?
dfs = []
# ss=[15]
ss = range(16)
# ss=[13,14]
for subject in ss:
    summary = pd.read_csv('second_Rawsummary' + str(subject) + '.csv')
    dfs.append(summary)
    # summary = summary[summary.repetition > 1]
    fs = summary.groupby([summary.target_type, summary.cursor_type]).mean()

    # fs= fs.reset_index()
    # print(subject, '\n', fs.success_trial * 100)
    # g = sns.catplot(kind='bar', data=fs.success_trial.reset_index(), col='target_type', x='cursor_type',
    #                 y='success_trial',order=['HEAD','NEWSPEED','NEWSTICKY','NEWSPEEDSTICKY'])
    # for ax in g.axes.ravel():
    #
    #     # add annotations
    #     for c in ax.containers:
    #         labels = [f'{(v.get_height()):.3f}' for v in c]
    #         ax.bar_label(c, labels=labels, label_type='edge')
    #     ax.margins(y=0.2)
    # plt.title(str(subject))
    # plt.show()
summary = pd.concat(dfs)
summary['attempt_time'] = summary.trial_time - summary.initial_contact_time
cut = 0.1
summary.loc[summary['straightness'] >= 1 - cut, 'curve'] = 'straight'
summary.loc[(summary['straightness'] < 1 - cut) & (summary['straightness'] > cut), 'curve'] = 'between'
summary.loc[summary['straightness'] <= cut, 'curve'] = 'curve'
errors = summary[summary.error.isna() == False]
# summary = summary[summary.error.isna() == True]
summary.success_trial = summary.success_trial * 100
summary = summary[summary.repetition > 1]
summary['modality'] = summary.cursor_type.apply(check_modality)
summary['inten'] = summary.cursor_type.apply(check_inten)
fs = summary.groupby([summary.cursor_type, summary.target_type]).mean()
fail_index = summary[summary.success_trial == 0].index
# summary['drop_count_count'] =
# summary.loc[fail_index]['drop_out_count']=None
summary['drop_out_count'].mask(summary.success_trial == 0,None,inplace=True)
summary['drop_count'].mask(summary.success_trial == 0,None,inplace=True)
bySubjects = summary.groupby([summary.subject_num, summary.cursor_type, summary.target_type]).mean()
bySubjects = bySubjects.reset_index()
bySubjects.to_csv("SecondStudySummary.csv")

print('Total', '\n', fs.success_trial * 100)
print(errors.error.groupby([errors.cursor_type, errors.target_type]).count())
g = sns.catplot(kind='bar', data=fs.success_trial.reset_index(), col='target_type', x='cursor_type', y='success_trial',
                order=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY']);
for ax in g.axes.ravel():

    # add annotations
    for c in ax.containers:
        labels = [f'{(v.get_height()):.3f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge')
    ax.margins(y=0.2)

plt.show()
print(summary.groupby(summary.curve).count().straightness
      )

# summary

bySubjects.loc[bySubjects['cursor_type'] == 'HEAD', 'cursor_type'] = 'Head'
bySubjects.loc[bySubjects['cursor_type'] == 'NEWSPEED', 'cursor_type'] = 'Speed'
bySubjects.loc[bySubjects['cursor_type'] == 'NEWSTICKY', 'cursor_type'] = 'Score'
bySubjects.loc[bySubjects['cursor_type'] == 'NEWSPEEDSTICKY', 'cursor_type'] = 'Speed+Score'
successful = summary[summary.success_trial == 1]

s_bysubjects = successful.groupby(
    [successful.subject_num, successful.modality, successful.inten, successful.target_type]).mean()
s_bysubjects = s_bysubjects.reset_index()
# s_bysubjects.loc[s_bysubjects['cursor_type'] == 'HEAD', 'cursor_type'] = 'Head'
# s_bysubjects.loc[s_bysubjects['cursor_type'] == 'NEWSPEED', 'cursor_type'] = 'Speed'
# s_bysubjects.loc[s_bysubjects['cursor_type'] == 'NEWSTICKY', 'cursor_type'] = 'Score'
# s_bysubjects.loc[s_bysubjects['cursor_type'] == 'NEWSPEEDSTICKY', 'cursor_type'] = 'Speed+Score'
# %%


# for column in [
#     # 'longest_dwell_time',
#     # 'drop_count',
#     # 'initial_contact_time',
#     # 'attempt_time',
#     # 'success_trial',
#     # 'mean_dwell_time',
#     # 'std_offset',
#     # 'mean_offset_horizontal',
#     # 'mean_offset_vertical',
#     # 'std_offset_horizontal',
#     #       'std_offset_vertical',
#     # 'trial_time',
#     'drop_count',
#     # 'out_mean_distance',
#     # 'estimated_width','estimated_height',
#     # 'drop_out_count',
#     # 'mean_out_time'
#     # 'abs_mean_offset_horizontal', 'abs_mean_offset_vertical'
# ]:
#     g = sns.catplot(kind='bar', data=bySubjects, x='target_type', hue='cursor_type',
#                     y=column,
#                     linewidth=2.5,
#                     ci='sd',
#                     # facecolor=(1, 1, 1, 0),
#                     errcolor=".2", edgecolor=".2",
#                     # hatch=['/','x','-','.'],
#                     hue_order=['Head', 'Speed', 'Score', 'Speed+Score'],
#                     # legend=False
#                     )
#     if column == 'success_trial':
#         g.axes[0, 0].set_ylabel("Success Rate (%)")
#         g.axes[0, 0].set_box_aspect(8 / len(g.axes[0, 0].patches))
#     g.axes[0, 0].set_ylabel("Count")
#     g.axes[0, 0].set_box_aspect(8 / len(g.axes[0, 0].patches))
#     for ax in g.axes.ravel():
#
#         # add annotations
#         for c in ax.containers:
#             labels = [f'{(v.get_height()):.1f}' for v in c]
#             # ax.bar_label(c, labels=labels, label_type='edge')
#             ax.bar_label(c, labels=labels, label_type='center', fontsize=7)
#         ax.margins(y=0.2)
#
#     # plt.legend(ncol=4,loc='best')
#     # plt.tight_layout()
#     plt.show()
# %%ANOVA
# successful.drop('cursor_type',inplace=True)
# summary = successful.groupby(
#     [successful.modality, successful.inten, successful.target_type]).mean()
summary = s_bysubjects
# summary=summary
for column in [
    # 'longest_dwell_time',
    # 'drop_count',
    # 'initial_contact_time',
    # 'attempt_time',
    'success_trial',
    # 'mean_dwell_time',
    # 'std_offset',
    # 'mean_offset_horizontal',
    # 'mean_offset_vertical',
    # 'std_offset_horizontal',
    #       'std_offset_vertical',
    # 'trial_time',
    # 'drop_count',
    # 'out_mean_distance',
    # 'estimated_width','estimated_height',
    # 'drop_out_count',
    # 'mean_out_time'
    # 'abs_mean_offset_horizontal', 'abs_mean_offset_vertical'
]:
    # aovrm = AnovaRM(summary, column, 'subject_num', within=['cursor_type', 'target_type'],
    #                 aggregate_func='mean').fit()

    print('METRIC', column)
    # ax = sns.boxplot(data=bySubjects, x='target_type', y=column, hue='cursor_type', showfliers=False, showmeans=True,
    #                  meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
    #                             'markersize': '10'},
    #                  palette='Set1', width=0.8)
    # # annot = Annotator(ax, pairs, data=bySubjects, x='posture', y=column, order=order, hue='cursor_type',
    # #                   hue_order=hue_order)
    # # annot.new_plot(ax, pairs, data=s, x='posture', y=column, order=order, hue='cursor_type', hue_order=hue_order)
    # # annot.configure(test='t-test_ind', comparisons_correction="Bonferroni").apply_test().annotate()
    # # sns.boxplot(data=ss, x='posture', y=column, hue='cursor_type', showfliers=False, showmeans=True, )
    # plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1), title='Cursor Type')
    # plt.title(column)
    # plt.tight_layout()
    # plt.show()
    from statsmodels.sandbox.stats.multicomp import MultiComparison
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    import scipy.stats
    from bioinfokit.analys import stat
    from statannotations.Annotator import Annotator

    # comp = MultiComparison(data=summary[column], groups=summary['cursor_type'])
    # result = comp.allpairtest(scipy.stats.ttest_ind, method='bonf')
    # res = stat()
    # res.anova_stat(df=summary, res_var=column,
    #                anova_model=column + ' ~ C(target_type)+C(cursor_type)+C(curve)+C(target_type):C(cursor_type)+C(target_type):C(curve)+C(cursor_type):C(curve)+C(target_type):C(cursor_type):C(curve)')
    # print(res.anova_summary)

    # res.tukey_hsd(df=summary, res_var=column, xfac_var='target_type',
    #               anova_model=column + ' ~ C(target_type)+C(cursor_type)+C(target_type):C(cursor_type)')
    #
    #
    # print(res.tukey_summary)
    # tukey_results = pairwise_tukeyhsd(endog=summary[column], groups=summary['target_type'])
    # aov.round(3)
    # print('RM-anova(statsmodel)')
    # print(aovrm)
    # ax = sns.boxplot(data=summary, x='target_type', y=column, hue='cursor_type',
    #                  # hue_order=['straight', 'between', 'curve'],
    #                  showfliers=False,
    #                  showmeans=False, )
    # pairs = []
    # for tt in ['GRID', 'MENU', 'PIE']:
    #     for ct in itertools.combinations(['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'], 2):
    #         pairs.append(((tt, ct[0]), (tt, ct[1])))
    #
    # # annotator = Annotator(ax, pairs, data=summary, x='target_type', hue='cursor_type',
    # #                       # hue_order=['straight', 'between', 'curve'],
    # #                       y=column)
    # # annotator.configure(test="t-test_ind", comparisons_correction='Bonferroni', verbose=False).apply_and_annotate()
    # plt.title(column)
    #
    # plt.show()
    # print('Posthoc(pairwise ttest,pg)')
    if False:
        for tt in ['GRID', 'MENU', 'PIE']:
            ax = sns.boxplot(data=summary[summary.target_type == tt], x='cursor_type', y=column, hue='curve',
                             hue_order=['straight', 'between', 'curve'],
                             showfliers=False,
                             showmeans=True, )
            pairs = [
                (('HEAD', 'straight'), ('HEAD', 'between')),
                (('HEAD', 'straight'), ('HEAD', 'curve')),
                (('HEAD', 'curve'), ('HEAD', 'between')),
                (('NEWSPEED', 'straight'), ('NEWSPEED', 'between')),
                (('NEWSPEED', 'straight'), ('NEWSPEED', 'curve')),
                (('NEWSPEED', 'curve'), ('NEWSPEED', 'between')),
                (('NEWSTICKY', 'straight'), ('NEWSTICKY', 'between')),
                (('NEWSTICKY', 'straight'), ('NEWSTICKY', 'curve')),
                (('NEWSTICKY', 'curve'), ('NEWSTICKY', 'between')),
                (('NEWSPEEDSTICKY', 'straight'), ('NEWSPEEDSTICKY', 'between')),
                (('NEWSPEEDSTICKY', 'straight'), ('NEWSPEEDSTICKY', 'curve')),
                (('NEWSPEEDSTICKY', 'curve'), ('NEWSPEEDSTICKY', 'between')),
            ]
            annotator = Annotator(ax, pairs, data=summary[summary.target_type == tt], x='cursor_type', hue='curve',
                                  hue_order=['straight', 'between', 'curve'],
                                  y=column)
            annotator.configure(test="t-test_ind", comparisons_correction='Bonferroni', verbose=2).apply_and_annotate()

            plt.title(str(tt))
            plt.show()
    # print(result[0])
    # aov = pg.rm_anova(dv=column, within=['cursor_type', 'target_type'],
    #                   subject='subject_num', data=summary, detailed=True,
    #                   effsize="np2", correction=True)
    # # sph = pg.sphericity(dv=column, within=['cursor_type','target_type'],
    # #                     subject='subject_num', data=summary,
    # #                     )
    print('anova(pg)')
    # pg.print_table(aov)
    anova = pg.anova(dv=column, between=['modality', 'inten', 'target_type'], effsize='np2', detailed=True,
                     data=s_bysubjects)
    pg.print_table(anova)
    print('anova(statsmodel)')
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    model = ols(column + """ ~C(modality) + C(inten) + C(target_type) +
    C(modality):C(inten) +C(modality):C(target_type) + C(inten):C(target_type)+
    C(modality):C(inten):C(target_type) 
    """, data=summary
                ).fit()

    table1 = sm.stats.anova_lm(model)
    print(table1.round(3))
    from statsmodels.graphics.factorplots import interaction_plot

    # fig = interaction_plot(x=summary['inten'], trace=summary['modality'], response=summary[column],
    #                        )
    sns.catplot(data=summary, x='target_type', col='modality', kind='bar', y=column, hue='inten', palette='Set3')
    plt.show()
    # fig = pg.plot_shift(x, y)
    for wt in ["GRID", 'MENU', 'PIE']:
        pg_posthoc = pg.pairwise_ttests(dv=column, within=['inten', 'modality'], subject='subject_num',
                                        data=summary[summary.target_type == wt],
                                        padjust='bonf')
        print(wt)

        pg.print_table(pg_posthoc)
    import scikit_posthocs as sp
    # for g in ['modality', 'inten', 'target_type']:
    #     comp = MultiComparison(data=summary[column], groups=summary[g])
    #     result = comp.allpairtest(scipy.stats.ttest_ind, method='bonf')
    #     print(result[0])
    #
    #     print(sp.posthoc_ttest(summary,val_col=column,group_col=g,p_adjust='holm'))
# %%
# subjects = [0, 1, 2, 3, 4, 5, 6]
subjects = range(16)
dfs = []
for subject in subjects:
    summary_subject = pd.read_csv('second_Rawsummary' + str(subject) + '.csv')
    dfs.append(summary_subject)
summary = pd.concat(dfs)
# summary=summary.append(summary2)
summary['estimated_width'] = summary.mean_offset_horizontal.apply(
    abs) + 2 * summary.std_offset_horizontal
summary['estimated_height'] = summary.mean_offset_vertical.apply(abs) + 2 * summary.std_offset_vertical
fs = summary.groupby([summary.target_type, summary.cursor_type]).mean()


# fs.reindex(['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'])

# sns.boxplot(data=summary, x='cursor_type', y='longest_dwell_time', showfliers=False,
#             showmeans=True,
#             meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
#                        'markersize': '10'}, hue='target_type')
# plt.show()
def avg(d):
    return sum(d) / len(d)


summary.out_mean_distance = summary.out_mean_distance.apply(np.array).apply(avg)
for c in [
    # 'longest_dwell_time',
    #       'initial_contact_time',
    #       'mean_offset',
    #       'std_offset', 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal',
    #       'std_offset_vertical',
    # 'trial_time', 'drop_count',
    'out_mean_distance',
    # 'estimated_width','estimated_height',
    # 'drop_out_count', 'mean_out_time'
    # 'abs_mean_offset_horizontal', 'abs_mean_offset_vertical'
]:
    sns.boxplot(data=summary, x='target_type', y=c, showfliers=False,
                showmeans=True,
                meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '10'}, hue='cursor_type',
                hue_order=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'])
    plt.show()
# %% STRAIGHTNESS

print(summary.groupby(summary.curve).count().straightness)
bySubjects = summary.groupby([summary.subject_num, summary.cursor_type, summary.target_type]).mean()
bySubjects = bySubjects.reset_index()
# print(len(summary), 'straight: ',
#       len(summary[summary.straightness == 1]), '\n curve: ',
#       len(summary[summary.straightness == 0]), '\n between:',
#       len(summary[(0 < summary.straightness) & (summary.straightness < 1)]),
#       )
successful = summary[summary.success_trial == 1]
for c in [
    # 'longest_dwell_time',
    # 'initial_contact_time',
    # 'mean_offset',
    # 'std_offset',
    # 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal',
    # 'std_offset_vertical',
    # 'trial_time',
    # 'drop_count',
    'success_trial'
    # 'out_mean_distance',
    # 'estimated_width', 'estimated_height',
    # 'drop_out_count', 'mean_out_time',
    # 'abs_mean_offset_horizontal', 'abs_mean_offset_vertical'
]:
    g = sns.catplot(kind='bar', data=summary, col='target_type', x='cursor_type',
                    y=c,
                    ci=None,
                    # order=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'],
                    # ci='sd',
                    # capsize=.2,
                    # palette='husl',
                    dodge=True,
                    # hue='curve'
                    )
    for ax in g.axes.ravel():

        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()):.1f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
    # sns.boxplot(data=summary, x='curve', y=c, showfliers=False,
    #             showmeans=True,
    #             meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
    #                        'markersize': '10'},
    #             hue='cursor_type',
    #             hue_order=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY']
    #             )
    plt.show()
# %% corr plot
summary.corrwith(summary['straightness'], method='spearman')
# sns.regplot(x= summary['straightness'],y=summary['drop_count'])
# plt.show()
# for tt in ['GRID', 'MENU', 'PIE']:
#     df = summary[summary.target_type == tt].drop(['Unnamed: 0', 'posture', 'target_num', 'error', 'success_trial'],
#                                                  axis=1).corr()
#     fig, ax = plt.subplots(figsize=(15, 15))
#     # mask = np.zeros_like(df, dtype=np.bool)
#     # mask[np.triu_indices_from(mask)] = True
#     sns.clustermap(df,
#                    annot=True,
#                    # mask=mask,
#                    linewidths=0.5,
#                    cmap='Greens',
#                    vmin=-1, vmax=1)
#     plt.tight_layout()
#     plt.show()

# %% drop position plot
from ast import literal_eval

for tt in ['GRID', 'MENU', 'PIE']:

    for i in range(8):
        df = summary[(summary.target_type == tt) & (summary.target_num == i)]
        xs = []
        ys = []
        d = 5.5
        for x in df.drop_positions:
            try:
                x = literal_eval(x)

                for c in x:
                    pos = np.array(c)
                    pos = pos / np.linalg.norm(pos)
                    xs.append(float(pos[0]))
                    ys.append(float(pos[1]))
            except Exception as e:
                pass
                # print(e.args)

        plt.axis('equal')
        if tt == "GRID":
            _X = (d / 2 * (i % 4 - 2) + 1) * 2
            _Y = 2 if i < 4 else -2
        elif tt == 'MENU':
            _X = 0
            _Y = d / 2 * (4 - i) - 1
        elif tt == "PIE":
            r2 = math.sqrt(2) / 2

            if i == 0:
                _X = 0
                _Y = d
            elif i == 1:
                _X = r2 * d
                _Y = r2 * d
            elif i == 2:
                _X = d
                _Y = 0
            elif i == 3:
                _X = r2 * d
                _Y = -r2 * d
            elif i == 4:
                _X = 0
                _Y = -d
            elif i == 5:
                _X = -r2 * d
                _Y = -r2 * d
            elif i == 6:
                _X = -d
                _Y = 0
            elif i == 7:
                _X = -r2 * d
                _Y = r2 * d

        plt.scatter(-pd.Series(xs) + _X, pd.Series(ys) + _Y, marker='.', alpha=0.01)
        print(tt, i, pd.Series(xs).apply(abs).mean(), pd.Series(ys).apply(abs).mean())
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title(tt + str(i))
    plt.show()
    # plt.scatter()

# %%draw ellipses
ss = draw_ellipse_second_study()
# %%
f = pd.read_pickle('secondOffsets.pkl')
f = f[f.error.isna()]
fig, ax = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True)
for idx, tt in enumerate(['GRID', 'MENU', 'PIE']):
    for cursor_type in ['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY']:

        h_raw = f[(f.cursor_type == cursor_type) & (f.target_type == tt)]['coordinates_x'].values
        v_raw = f[(f.cursor_type == cursor_type) & (f.target_type == tt)]['coordinates_y'].values
        h = []
        v = []
        for i in range(len(h_raw)):
            h += h_raw[i]
            v += v_raw[i]
        h = np.array(h)
        v = np.array(v)
        if cursor_type == 'HEAD':
            ec = 'red'
            ap = 0.03
        elif cursor_type == 'NEWSPEED':
            ec = 'green'
            ap = 0.05
        elif cursor_type == 'NEWSTICKY':
            ec = 'blue'
            ap = 0.05
        elif cursor_type == 'NEWSPEEDSTICKY':
            ec = 'cyan'
            ap = 0.05
        ax[idx].scatter(h, v, s=0.1, alpha=ap, c=ec)
        plt_confidence_ellipse(h, v, ax[idx], 2, linestyle='dotted', facecolor='None', edgecolor=ec,
                               linewidth=3)
        # print(
        #     ax[idx].patches)
        ax[idx].set_title(str(cursor_type).lower())
        ax[idx].set_xlim(-12, 12)
        ax[idx].set_ylim(-6, 6)
        ax[idx].set_aspect('equal')
        ax[idx].grid()
plt.show()

# %%
sub_num = 13
ct = 'HEAD'
# rep = 3
tt = 'GRID'
# t=3
dfs = []
for t in range(8):
    for rep in [4]:
        temp_data = read_second_data(subject=sub_num, cursor_type=ct, repetition=rep, target_type=tt, target_num=t)
        dfs.append(temp_data)
data = pd.concat(dfs)

from sklearn.decomposition import PCA


def arrow(v1, v2, ax):
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate("", v2, v1, arrowprops=arrowprops)


# import mglearn
pca = PCA(n_components=2)
X = data[['head_position_x', 'head_position_z']]
# X=X.T
X = np.array(X)
# plt.axis('equal')
# plt.scatter(X[:,0], X[:,1]);
# plt.show()
pca.fit(X)
print("Principal axes:", pca.components_)
print("Explained variance:", pca.explained_variance_)
print("Mean:", pca.mean_)
Z = pca.transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].axis('equal')
axes[0].scatter(X[:, 0], X[:, 1])
axes[1].axis('equal')
axes[1].set_xlim(-3, 3)
axes[1].scatter(Z[:, 0], Z[:, 1])
for l, v in zip(pca.explained_variance_, pca.components_):
    arrow([0, 0], v * l * 3, axes[0])
for l, v in zip([1.0, 0.16], [np.array([1.0, 0.0]), np.array([0.0, 1.0])]):
    arrow([0, 0], v * l * 3, axes[1])
axes[0].set_title("Original")
axes[1].set_title("Transformed");
ZZ = pd.DataFrame(Z)
ZM = ZZ[ZZ[0] < 0]
ZP = ZZ[ZZ[0] > 0]
leftMax = ZM[0].values[ZM[1].argmax()]
leftMin = ZM[0].values[ZM[1].argmin()]
lefty = max(leftMax, leftMin)
rightMax = ZP[0].values[ZP[1].argmax()]
rightMin = ZP[0].values[ZP[1].argmin()]
righty = min(rightMin, rightMax)
plt.axis('equal')
for x in [lefty, righty]:
    axes[1].axvline(x)
plt.tight_layout()
plt.show()
for i in range(8):
    temp_data = data[data.end_num == i]
    temp_X = temp_data[['head_position_x', 'head_position_z']]
    temp_X = np.array(temp_X)
    temp_transform = pca.transform(temp_X)
    straightness = np.sum((lefty < temp_transform[:, 0]) & (temp_transform[:, 0] < righty)) / len(temp_transform)
    plt.scatter(temp_transform[:, 0], temp_transform[:, 1])
    print(i, straightness)

plt.axis('equal')
for x in [lefty, righty]:
    plt.axvline(x)
plt.show()

# %% SUBJECTIVE
subjective = pd.read_csv('Subjective_study2.csv')
means = subjective.groupby([subjective.mm,subjective.tr,subjective.target_type]).mean().round(2)
means.insert(len(means.columns)-1, 'nasa', means.pop('nasa'))
means.insert(len(means.columns)-1, 'borg', means.pop('borg'))
# aov = pg.rm_anova(dv='borg', within=['mm', 'tr'],
#                   subject='subject_num', data=subjective, detailed=True,
#                   effsize="np2", correction=True)
# # aov = pg.anova(dv='borg',between=['posture','cursor_type'],data=subjective,detailed=True)
# print(aov.round(3))
# ph = pg.pairwise_ttests(data=subjective, dv='borg', between='mm')
# print(ph.round(3))
# ph = pg.pairwise_ttests(data=subjective, dv='borg', between='tr')
# print(ph.round(3))
# aov = pg.rm_anova(dv='nasa', within=['mm', 'tr'],
#                   subject='subject_num', data=subjective, detailed=True,
#                   effsize="np2", correction=True)
# # aov = pg.anova(dv='borg',between=['posture','cursor_type'],data=subjective,detailed=True)
# print(aov.round(3))
# ph = pg.pairwise_ttests(data=subjective, dv='borg', between='mm')
# print(ph.round(3))
# ph = pg.pairwise_ttests(data=subjective, dv='borg', between='tr')
# print(ph.round(3))

# %% walklength check?
# print(summary.groupby([summary.subject_num]).walking_speed.mean())
temp_data = read_second_data(subject=10, cursor_type='HEAD', repetition=3, target_type='GRID', target_num=2)
walklength = (temp_data.head_position_x.diff(1) ** 2 + temp_data.head_position_z.diff(1) ** 2).apply(
    math.sqrt).sum()
trial_time = temp_data.timestamp.values[-1] - temp_data.timestamp.values[0]
print(walklength / trial_time)
walklength = (temp_data.head_position_x.diff(1) ** 2 + temp_data.head_position_z.diff(1) ** 2).apply(
    math.sqrt) / temp_data.timestamp.diff(1)
print(walklength.mean())
