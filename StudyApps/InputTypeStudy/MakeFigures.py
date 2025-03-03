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
import itertools
from scipy.stats import anderson
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import numpy as np

# pio.renderers.default = 'browser'
# pd.set_option('mode.chained_assignment', None)
# sns.set_style('ticks')
# sns.set_context('talk')
by_subject = pd.read_csv("newstudy_BySubject.csv")
summary1 = pd.read_csv("newstudy_summary.csv")

only_success = summary1[summary1.success > 99]
by_subject_success_only = only_success.groupby(
    [only_success.subject, only_success.selection, only_success.posture, only_success.cursor]).mean()
by_subject_success_only = by_subject_success_only.reset_index()
# %% Primary Figures
import patchworklib as pw

custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(context='paper',  # 매체: paper, talk, poster
              style='whitegrid',  # 기본 내장 테마
              # palette='deep',       # 그래프 색
              font_scale=7,  # 글꼴 크기
              rc=custom_params)  # 그래프 세부 사항


pw.overwrite_axisgrid()
data = by_subject.copy()
data.loc[data.posture == "Walk", 'posture'] = 'Circuit'
data.rename(columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"}, inplace=True)

# g= sns.catplot(kind='bar',data=data, col='Trigger', x='Mobility', y='error_trial', hue='Modality',
#             order=['Stand', 'Treadmill', 'Circuit'],
#             hue_order=['Eye', 'Head', 'Hand'], capsize=.05, errwidth=2, linewidth=5.0,
#             palette='muted')
# g=pw.load_seaborngrid(g,figsize=(18,18))
# g.move_legend(new_loc="upper left",bbox_to_anchor=(.1, 0.85))

g = sns.FacetGrid(data, col='Trigger', height=18, aspect=0.65)
g.map_dataframe(sns.barplot,
                x='Mobility', y='error_trial', hue='Modality', order=['Stand', 'Treadmill', 'Circuit'],
                hue_order=['Eye', 'Head', 'Hand'], capsize=.05, errwidth=2, linewidth=5.0,
                palette='muted')

axes = g.axes.flatten()
axes[0].set_title("Click")
axes[1].set_title("Dwell")

g.set(ylim=(0, 100), yticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
g.set_axis_labels("", "Error Rate " + '(%)')
# g.axes[0,0].legend()
g = pw.load_seaborngrid(g, label="g")
# g.move_legend("upper left", bbox_to_anchor=(.1, 0.85), ncol=1, frameon=True)

data = by_subject_success_only.copy()
data.loc[data.posture == "Walk", 'posture'] = 'Circuit'
data.rename(columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"}, inplace=True)
gt = sns.FacetGrid(data, col="Trigger", height=18, aspect=.65)
gt.map_dataframe(sns.barplot, x="Mobility", y="overall_time", hue="Modality", hue_order=['Eye', 'Head', 'Hand'],
                 palette="pastel", capsize=.05, errwidth=2, linewidth=5.0, errcolor=".5",
                 )
gt.map_dataframe(sns.barplot, x="Mobility", y="initial_contact_time", hue="Modality", hue_order=['Eye', 'Head', 'Hand'],
                 palette="muted",
                 capsize=.05, errwidth=2, linewidth=5.0,
                 )
axes = gt.axes.flatten()
axes[0].set_title("Click")
axes[1].set_title("Dwell")
# Note: the default legend is not resulting in the correct entries.
#       Some fix-up step is required here...
gt.set_axis_labels("", 'Time (s)')
gt = pw.load_seaborngrid(gt, label="gt")

data = by_subject.copy()
data.loc[data.posture == "Walk", 'posture'] = 'Circuit'
data.rename(columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"}, inplace=True)
gi = sns.FacetGrid(data, height=18, aspect=.38)
with plt.rc_context({'lines.linewidth': 3.0}):
    gi.map_dataframe(sns.pointplot, x='Trigger', y="error_trial", hue='Mobility', ci=None, scale=1.5,
                     markers=['o', 'x', 's'],linestyles=['-', '-.', ':'], palette='gray')
gi.set_axis_labels("", "Error Rate (%)")
gi.set(ylim=(0, 100))
gi = pw.load_seaborngrid(gi, label='gi')

git = sns.FacetGrid(data, height=18, aspect=.38)
with plt.rc_context({'lines.linewidth': 3.0}):
    git.map_dataframe(sns.pointplot, x='Trigger', y="dwelling_time", hue='Mobility', ci=None, scale=1.5,
                      markers=['o', 'x', 's'],linestyles=['-', '-.', ':'], palette='gray')
git.set_axis_labels("", "Completion Time (s)")
git.set(ylim=(0, 4))
git = pw.load_seaborngrid(git, label='git')
pw.param["margin"] = 3.0
g.case.set_title("(a)", x=0.5, y=-0.1, loc='center')
gi.case.set_title("(b)", x=0.5, y=-0.1, loc='center')
gt.case.set_title("(c)", x=0.5, y=-0.1, loc='center')
git.case.set_title("(d)", x=0.5, y=-0.1, loc='center')
(gi | g | git | gi).outline.savefig("PrimaryResults.pdf")

# %% Collapsed first figure - entry + speed + width/height_ dwell
data = summary1.copy()
data.loc[data.posture == "Walk", 'posture'] = 'Circuit'
data.rename(columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"}, inplace=True)
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(context='paper',  # 매체: paper, talk, poster
              style='whitegrid',  # 기본 내장 테마
              # palette='deep',       # 그래프 색
              font_scale=4,  # 글꼴 크기
              rc=custom_params)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=4, sharey=False, figsize=(40, 18))
for i, title in enumerate(['target_in_count', 'mean_cursor_speed']):
    hcut = 0.9772
    lcut = 0.0228
    qhh = data[title].quantile(hcut)
    qhl = data[title].quantile(lcut)
    d = data[(data[title] < qhh) & (data[title] > qhl) & (data.success > 99)]
    if (title=='target_in_count'): d=data.copy()
    by_subject1 = d.groupby([d.subject, d.Trigger, d.Mobility, d.Modality]).mean()
    by_subject1 = by_subject1.reset_index()
    sns.boxplot(data=by_subject1[by_subject1.Trigger == "Dwell"],
                # col='selection',
                x='Mobility', y=title,
                hue='Modality', palette='muted',

                order=['Stand', 'Treadmill', 'Circuit'], hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                # boxprops=dict( edgecolor='gray'),
                width=0.5,
                dodge=True,
                ax=axes[i]
                )
for i, title in enumerate(['width', 'height']):
    hcut = 0.9772
    lcut = 0.0228
    # if title=='height':
    #     hcut=0.95
    if title == 'width':
        hcut = 0.95
    qhh = data[title].quantile(hcut)
    qhl = data[title].quantile(lcut)
    # print(qhh)
    d = data[(data[title] < qhh) & (data[title] > qhl)]
    by_subject1 = d.groupby([d.subject, d.Trigger, d.Mobility, d.Modality]).mean()
    by_subject1 = by_subject1.reset_index()

    # d = d[(d.final_point_horizontal < qhh) & (d.final_point_horizontal > qlh) & (d.final_point_vertical < qhv) & (
    #         d.final_point_vertical > qlv)]
    # plt.rcParams['figure.constrained_layout.use'] = True
    sns.boxplot(data=by_subject1[by_subject1.Trigger == "Dwell"],
                # col='selection',
                x='Mobility', y=title,
                hue='Modality', palette='muted',

                order=['Stand', 'Treadmill', 'Circuit'], hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                # boxprops=dict( edgecolor='gray'),
                width=0.5,
                dodge=True,
                ax=axes[i + 2]
                )
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[0].set_title("Target Entries")
axes[1].set_title("Average Cursor Speed")
axes[0].set_ylabel("Target Entries (Count)")
axes[1].set_ylabel("Cursor Speed (deg/s)")
axes[1].yaxis.set_tick_params(which='both', labelbottom=True)
axes[0].set_yticks(ticks=range(11))
axes[1].set_yticks(ticks=range(0,25,3))
axes[2].get_legend().remove()
# axes[3].get_legend().remove()
axes[2].set_title("Width")
axes[3].set_title("Height")
axes[2].set_ylim((0, 8))
axes[3].set_ylim((0, 8))
axes[2].set_ylabel("Width (deg)")
axes[3].set_ylabel("Height (deg)")
axes[2].yaxis.set_tick_params(which='both', labelbottom=True)
axes[3].yaxis.set_tick_params(which='both', labelbottom=True)
# plt.legend(loc='upper left',
#            bbox_to_anchor=(0.5, 1.05),
#                fancybox=True, ncol=3)
# axes[0].legend(
#               # loc='center right'
#               loc='best'
#               )
axes[0].set_ylim((0, 11))
axes[1].set_ylim((0, 25))
axes[2].set_ylim((0, 8))
axes[3].set_ylim((0, 8))
sns.move_legend(axes[3], "upper center",
                bbox_to_anchor=(.8, .95), ncol=1, title="Modality", frameon=True, )
plt.tight_layout()
plt.show()
# plt.savefig("Collapsed_Dwell.pdf")

# %% Collapsed first figure - entry + speed + width/height_ click

data = summary1.copy()
data.loc[data.posture == "Walk", 'posture'] = 'Circuit'
data.rename(columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"}, inplace=True)
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(context='paper',  # 매체: paper, talk, poster
              style='whitegrid',  # 기본 내장 테마
              # palette='deep',       # 그래프 색
              font_scale=4,  # 글꼴 크기
              rc=custom_params)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=4, sharey=False, figsize=(40, 18))
# title = 'target_in_count_per_second'
for i, title in enumerate(['target_in_count', 'mean_cursor_speed']):
    hcut = 0.9772
    lcut = 0.0228
    qhh = data[title].quantile(hcut)
    qhl = data[title].quantile(lcut)

    d = data[(data[title] < qhh) & (data[title] > qhl) & (data.success > 99)]
    if (title == 'target_in_count'): d = data.copy()
    by_subject1 = d.groupby([d.subject, d.Trigger, d.Mobility, d.Modality]).mean()
    by_subject1 = by_subject1.reset_index()
    sns.boxplot(data=by_subject1[by_subject1.Trigger == "Click"],
                # col='selection',
                x='Mobility', y=title,
                hue='Modality', palette='muted',

                order=['Stand', 'Treadmill', 'Circuit'], hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                # boxprops=dict( edgecolor='gray'),
                width=0.5,
                dodge=True,
                ax=axes[i]
                )
d = summary1.copy()
d.loc[d.posture == "Walk", 'posture'] = 'Circuit'
d.rename(columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"}, inplace=True)
hcut = 0.9772
lcut = 0.0228
qlh = d['final_point_horizontal'].quantile(lcut)
qhh = d['final_point_horizontal'].quantile(hcut)
qlv = d['final_point_vertical'].quantile(lcut)
qhv = d['final_point_vertical'].quantile(hcut)
d = d[(d.final_point_horizontal < qhh) & (d.final_point_horizontal > qlh) & (d.final_point_vertical < qhv) & (
        d.final_point_vertical > qlv)]
# sh = d.final_point_horizontal
# sv = d.final_point_vertical
hm = d.groupby([d.Mobility, d.Trigger, d.Modality]).final_point_horizontal.mean().apply(abs)
hs = d.groupby([d.Mobility, d.Trigger, d.Modality]).final_point_horizontal.std()
vm = d.groupby([d.Mobility, d.Trigger, d.Modality]).final_point_vertical.mean().apply(abs)
vs = d.groupby([d.Mobility, d.Trigger, d.Modality]).final_point_vertical.std()
width_df = hm + 2 * hs
width_df = width_df.reset_index()
height_df = vm + 2 * vs
height_df = height_df.reset_index()
dm = d.groupby([d.subject, d.Mobility, d.Trigger, d.Modality]).mean().apply(abs)
ds = d.groupby([d.subject, d.Mobility, d.Trigger, d.Modality]).std()
dd = dm + 2 * ds
dd = dd.reset_index()
sns.boxplot(data=dd,
            # col='selection',
            x='Mobility', y='final_point_horizontal',
            hue='Modality', palette='muted',
            order=['Stand', 'Treadmill', 'Circuit'], hue_order=['Eye', 'Head', 'Hand'],
            showfliers=False, showmeans=True,
            meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                       'markersize': '15'},

            # boxprops=dict( edgecolor='gray'),
            width=0.5,
            dodge=True,
            ax=axes[2]
            )
sns.boxplot(data=dd,
            # col='selection',
            x='Mobility', y='final_point_vertical',
            hue='Modality', palette='muted',
            order=['Stand', 'Treadmill', 'Circuit'], hue_order=['Eye', 'Head', 'Hand'],
            showfliers=False, showmeans=True,
            meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                       'markersize': '15'},

            # boxprops=dict( edgecolor='gray'),
            width=0.5,
            dodge=True,
            ax=axes[3]
            )
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[0].set_yticks(ticks=range(11))
axes[1].set_yticks(ticks=range(0,25,3))
axes[0].set_title("Target Entries")
axes[1].set_title("Average Cursor Speed")
axes[0].set_ylabel("Target Entries (Count)")
axes[1].set_ylabel("Cursor Speed (deg/s)")
axes[1].yaxis.set_tick_params(which='both', labelbottom=True)
axes[2].get_legend().remove()
# axes[3].get_legend().remove()
axes[2].set_title("Width")
axes[3].set_title("Height")
axes[2].set_ylabel("Width (deg)")
axes[3].set_ylabel("Height (deg)")
axes[0].set_ylim((0, 11))
axes[1].set_ylim((0, 25))
axes[2].set_ylim((0, 8))
axes[3].set_ylim((0, 8))
sns.move_legend(axes[3], "upper center",
                bbox_to_anchor=(.8, .95), ncol=1, title="Modality", frameon=True, )
plt.tight_layout()
# plt.show()
plt.savefig("Collapsed_Click.pdf")

# %% Threshold plots
dwell_threshold_summary1 = pd.read_csv("dwell_threshold_summary1.csv")
threshold_by_subject = pd.read_csv("threshold_by_subject.csv")


def change(t):
    if t == "True":
        return 1
    else:
        return 0


dwell_threshold_summary1["success"] = dwell_threshold_summary1["success"].apply(change)
dwell_threshold_summary1['width'] = dwell_threshold_summary1.mean_error_horizontal.apply(
    abs) + 2 * dwell_threshold_summary1.std_error_horizontal
dwell_threshold_summary1['height'] = dwell_threshold_summary1.mean_error_vertical.apply(
    abs) + 2 * dwell_threshold_summary1.std_error_vertical
threshold_by_subject = dwell_threshold_summary1.groupby(
    [dwell_threshold_summary1.subject, dwell_threshold_summary1.posture, dwell_threshold_summary1.selection,
     dwell_threshold_summary1.cursor,
     dwell_threshold_summary1.threshold]).mean()
threshold_by_subject = threshold_by_subject.reset_index()
dwell_threshold_summary_success_only = dwell_threshold_summary1[(dwell_threshold_summary1.success == 1)]
dwell_threshold_summary_success_only.drop("success", axis=1, inplace=True)
threshold_by_subject_success_only = dwell_threshold_summary_success_only.groupby(
    [dwell_threshold_summary_success_only.subject, dwell_threshold_summary_success_only.posture,
     dwell_threshold_summary_success_only.selection,
     dwell_threshold_summary_success_only.cursor,
     dwell_threshold_summary_success_only.threshold]).mean()
threshold_by_subject_success_only = threshold_by_subject_success_only.reset_index()
threshold_by_subject['Error_Rate'] = (1 - threshold_by_subject['success']) * 100
from matplotlib.patches import PathPatch


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    ##iterating through Axes instances
    for ax in g.axes.flatten():

        ##iterating through axes artists:
        for c in ax.get_children():

            ##searching for PathPatches
            if isinstance(c, PathPatch):
                ##getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                ##setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                ##setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


# %%
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(context='paper',  # 매체: paper, talk, poster
              style='whitegrid',  # 기본 내장 테마
              # palette='deep',       # 그래프 색
              font_scale=6,  # 글꼴 크기
              rc=custom_params)  # 그래프 세부 사항
pw.overwrite_axisgrid()
g = sns.catplot(kind='box', data=threshold_by_subject,
                col='posture',
                x='threshold', y='Error_Rate',
                hue='cursor',

                # order=['Stand', 'Treadmill', 'Walk'],
                hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                whiskerprops={'linestyle': '--'},
                width=0.8,
                # whis=1.0,
                dodge=True,
                height=18, aspect=1.2,legend=False
                )
# sns.move_legend(g, "upper center",
#                 bbox_to_anchor=(.1, .9), ncol=1, title="Modality", frameon=True, )
axes = g.axes.flatten()
axes[0].set_title("Stand")
axes[1].set_title("Treadmill")
axes[2].set_title("Circuit")
# axes[0].get_legend().remove()
# axes[1].get_legend().remove()
# axes[2].get_legend().remove()

g.set_axis_labels("", 'Error Rate (%)')
axes[0].set_ylabel('Error Rate (%)', fontsize=75)
adjust_box_widths(g, 0.8)
g = pw.load_seaborngrid(g, label="g")
threshold_by_subject['trial_duration'] = threshold_by_subject['first_dwell_time'] - threshold_by_subject_success_only[
    'initial_contact_time']
title = 'trial_duration'
data = threshold_by_subject.copy()
g1 = sns.catplot(kind='box', data=data,
                col='posture',
                x='threshold', y=title,
                hue='cursor',

                # order=['Stand', 'Treadmill', 'Walk'],
                hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                whiskerprops={'linestyle': '--'},
                width=0.8,
                # whis=1.0,
                dodge=True,
                height=18, aspect=1.2,legend=False
                )
# sns.move_legend(g1, "upper center",
#                 bbox_to_anchor=(.1, .9), ncol=1, title="Modality", frameon=True, )
axes = g1.axes.flatten()
axes[0].set_title("Stand")
axes[1].set_title("Treadmill")
axes[2].set_title("Circuit")
g1.set_axis_labels("", title)
if title=='trial_duration':ylabel= "Trigger Time (s)"
elif title=='target_in_count': ylabel='Target Entries (count)'
elif title=='final_cursor_speed': ylabel = 'Final Cursor Speed (deg/s)'

axes[0].set_ylabel(ylabel, fontsize=75)
adjust_box_widths(g1, 0.8)
g1 = pw.load_seaborngrid(g1, label="g1")

title = 'target_in_count'
data = threshold_by_subject.copy()
g2 = sns.catplot(kind='box', data=data,
                col='posture',
                x='threshold', y=title,
                hue='cursor',

                # order=['Stand', 'Treadmill', 'Walk'],
                hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                whiskerprops={'linestyle': '--'},
                width=0.8,
                # whis=1.0,
                dodge=True,
                height=18, aspect=1.2,legend=False
                )
# sns.move_legend(g2, "upper center",
#                 bbox_to_anchor=(.1, .9), ncol=1, title="Modality", frameon=True, )
axes = g2.axes.flatten()
axes[0].set_title("Stand")
axes[1].set_title("Treadmill")
axes[2].set_title("Circuit")
g2.set_axis_labels("", title)
if title=='first_dwell_time':ylabel= "Trigger Time (s)"
elif title=='target_in_count': ylabel='Target Entries (count)'
elif title=='final_cursor_speed': ylabel = 'Final Cursor Speed (deg/s)'

axes[0].set_ylabel(ylabel, fontsize=75)
adjust_box_widths(g2, 0.8)
g2 = pw.load_seaborngrid(g2, label="g2")

title = 'final_cursor_speed'
data = threshold_by_subject.copy()
g3 = sns.catplot(kind='box', data=data,
                col='posture',
                x='threshold', y=title,
                hue='cursor',

                # order=['Stand', 'Treadmill', 'Walk'],
                hue_order=['Eye', 'Head', 'Hand'],
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
                           'markersize': '15'},

                whiskerprops={'linestyle': '--'},
                width=0.8,
                # whis=1.0,
                dodge=True,
                height=18, aspect=1.2,legend=False
                )
# sns.move_legend(g3, "upper center",
#                 bbox_to_anchor=(.1, .9), ncol=1, title="Modality", frameon=True, )
axes = g3.axes.flatten()
axes[0].set_title("Stand")
axes[1].set_title("Treadmill")
axes[2].set_title("Circuit")
g3.set_axis_labels("", title)
if title=='first_dwell_time':ylabel= "Trigger Time (s)"
elif title=='target_in_count': ylabel='Target Entries (count)'
elif title=='final_cursor_speed': ylabel = 'Final Cursor Speed (deg/s)'

axes[0].set_ylabel(ylabel, fontsize=75)
adjust_box_widths(g3, 0.8)
g3 = pw.load_seaborngrid(g3, label="g3")


# for i, title in enumerate([
#     'first_dwell_time',
#     'target_in_count',
#     'final_cursor_speed',
# ]):
#     data = threshold_by_subject.copy()
#     g[i] = sns.catplot(kind='box', data=data,
#                     col='posture',
#                     x='threshold', y=title,
#                     hue='cursor',
#
#                     # order=['Stand', 'Treadmill', 'Walk'],
#                     hue_order=['Eye', 'Head', 'Hand'],
#                     showfliers=False, showmeans=True,
#                     meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
#                                'markersize': '15'},
#
#                     whiskerprops={'linestyle': '--'},
#                     width=0.8,
#                     # whis=1.0,
#                     dodge=True,
#                     height=18, aspect=1.2
#                     )
#     sns.move_legend(g[i], "upper center",
#                     bbox_to_anchor=(.1, .9), ncol=1, title="Modality", frameon=True, )
#     axes = g[i].axes.flatten()
#     axes[0].set_title("Stand")
#     axes[1].set_title("Treadmill")
#     axes[2].set_title("Circuit")
#     g[i].set_axis_labels("", title)
#     if title=='first_dwell_time':ylabel= "Trigger Time (s)"
#     elif title=='target_in_count': ylabel='Target Entries (count)'
#     elif title=='final_cursor_speed': ylabel = 'Final Cursor Speed (deg/s)'
#
#     axes[0].set_ylabel(ylabel, fontsize=75)
#     adjust_box_widths(g[i], 0.8)
#     g[i] = pw.load_seaborngrid(g[i], label="g"+str(i))
# g.case.set_title("A", x=0.1, y=-0.1, loc='center')
# g1.case.set_title("B", x=0.1, y=-0.9, loc='center')
# g2.case.set_title("C", x=0.5, y=-0.1, loc='center')
# g3.case.set_title("D", x=0.5, y=-0.1, loc='center')
(g / g1 / g2 / g3).outline.savefig("Thresholds.pdf")
