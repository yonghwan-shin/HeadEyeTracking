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

# from scipy.spatial.transform import Rotation as R
# pio.renderers.default = "browser"
pio.renderers.default = "notebook"
pd.set_option("mode.chained_assignment", None)
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

#%%
d, success_record = read_data(4, 3, 'Head', "Dwell", "Walk")
data = split_target(d)
t=3
temp_data = data[t].reset_index()
temp_data.timestamp -= temp_data.timestamp.values[0]
temp_data["EH_distance"] = (
    (temp_data.head_horizontal_offset - temp_data.eyeRay_horizontal_offset)
    ** 2
    + (temp_data.head_vertical_offset - temp_data.eyeRay_vertical_offset)
    ** 2
).apply(math.sqrt)
plt.plot(temp_data.timestamp.values,temp_data.horizontal_offset.values)
# plt.plot(temp_data.timestamp.values,temp_data.eyeRay_horizontal_offset.values)
plt.plot(temp_data.timestamp.values,temp_data.EH_distance.values)
plt.axhline(0)
plt.show()
# %%TEST
limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# bounds=[1,2,3,4,5,6,7,8,9,10]
# limits = [5,10]
bounds = [500]

for limit in limits:
    results = []
    for bound in bounds:
        d = speed_reduction_simulation_subject(range(24), limit, bound)
        results.append(d)

    simulation_summary = pd.concat(results)
    simulation_summary.to_csv("MAX_simulation_summary_" + str(limit) + ".csv")
# summary1[(summary1.subject==3)&(summary1.repetition==1)&(summary1.cursor=="Eye")&(summary1.selection=='Click')&(summary1.posture=="Stand")]
# %% Real test
# limits = [5,10]
limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
data = []
for limit in limits:
    # d = pd.read_csv("simulation_summary_"+str(limit) +".csv")
    d = pd.read_csv("MAX_simulation_summary_" + str(limit) + ".csv")
    data.append(d)
data = pd.concat(data)
# data.drop(['posture', 'selection'], axis=1, inplace=True)
# summary = data.groupby([data.speed_limit,data.bound]).mean()
# data.to_csv("MAX_simulation_summary_groupby.csv")
data.to_pickle("MAX_simulation_summary_groupby.pkl")
# %%
summary = pd.read_pickle("MAX_simulation_summary_groupby.pkl")
# summary = pd.read_csv("MAX_simulation_summary_groupby.csv")
by_subject = pd.read_csv("newstudy_BySubject.csv")

plotdata = summary.groupby(
    [
        summary.subject,
        summary.posture,
        summary.cursor,
        summary.selection,
        summary.speed_limit,
    ]
)["success"].mean()
plotdata = plotdata.reset_index()
plotdata["error_rate"] = 1 - plotdata["success"]
plotdata.loc[plotdata.posture == "Walk", "posture"] = "Circuit"
plotdata.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)

plotdata["error_rate"] = plotdata["error_rate"] * 100
by_subject.loc[by_subject.posture == "Walk", "posture"] = "Circuit"
by_subject.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
by_subject = by_subject[by_subject.Mobility != "Stand"]
by_subject["error_rate"] = 100 - by_subject.success
d = by_subject[["subject", "Modality", "Mobility", "Trigger", "success", "error_rate"]]
d["speed_limit"] = 21
plotdata = pd.concat([plotdata, d])
# g.map_dataframe(sns.barplot,
#                 x='Mobility', y='error_trial', hue='Modality', order=['Stand', 'Treadmill', 'Circuit'],
#                 hue_order=['Eye', 'Head', 'Hand'], capsize=.05, errwidth=2, linewidth=5.0,errcolor=".5",
#                 palette='pastel')

custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=4,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항


for sel in ["Click", "Dwell"]:
    full = plotdata.copy()
    plotdata = plotdata[plotdata.speed_limit <21]
    summary = plotdata[plotdata.Trigger == sel]
    plot_df = summary.groupby([summary.speed_limit,summary.Modality,summary.Mobility,summary.Trigger]).mean().reset_index()
    fig = px.line(plot_df, x='speed_limit', y='error_rate', color='Modality',   facet_row='Trigger',   facet_col='Mobility',
                markers=True,
                 labels={'error_rate': 'Error Rate (%)', 'speed_limit': 'Velocity Limit (°/s)'},
                #  title='error rates by rate, Posture, and Cursor',
                category_orders={'Mobility': ['Treadmill','Circuit'], "Modality": ['Eye','Head','Hand']},template='plotly_white',
                 symbol='Modality'
                 )
    fig.update_layout(height=300,width=900)
    colors=['Blue','Red','Green']
    if sel=='Click':
        row=1;col=1
        for i,n in enumerate([21.97,21.3,16.3 ]):
            fig.add_shape(type="line",x0=0, x1=20, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        row=1;col=2
        for i,n in enumerate([28.52, 30.91, 32.32]):
            fig.add_shape(type="line",x0=0, x1=20, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
    elif sel =='Dwell':
        row=1;col=1
        for i,n in enumerate([5.90,9.25,20.6]):
            fig.add_shape(type="line",x0=0, x1=20, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
        row=1;col=2
        for i,n in enumerate([37.13,53.59,55.82]):
            fig.add_shape(type="line",x0=0, x1=20, y0=n, y1=n,line=dict(dash='dot',color=colors[i], width=2),row=row,col=col)
    fig.update_yaxes(range=[0,60])
    fig.update_layout(showlegend=False)
    fig.update_traces(marker=dict(size=5))
    # fig.update_layout(
    #     title= f'{sel}: Error rates by rate, Posture, and Cursor',
    #     # xaxis_title='Window',
    #     # yaxis_title='Mean Measurement'
    # )
    # fig.show()
    fig.write_image(str(sel)+"speedlimit.png")
    # Dwell
    # # Treadmill
    # [5.90,9.25,20.6]
    # # circuit
    # [37.13,53.59,55.82]

    # Save the plot as an HTML file with inline data
    # fig.write_html(f"{sel}SpeedLimit.html", include_plotlyjs='cdn')

    # fig = sns.catplot(
    #     data=plotdata[plotdata.Trigger == sel],
    #     x="speed_limit",
    #     # y='initial_contact_time',
    #     y="error_rate",
    #     hue="Modality",
    #     col="Mobility",
    #     kind="bar",
    #     # showfliers=False, showmeans=True,
    #     # meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
    #     #         'markersize': '15'},
    #     hue_order=["Eye", "Head", "Hand"],
    #     # whiskerprops={'linestyle': '--'},
    #     width=0.8,
    #     # dodge=True,
    #     height=18,
    #     aspect=1.2,
    #     legend_out=False,
    #     # height=18, aspect=40/18
    # )
    
    # fig.set(ylim=(0, 100), yticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # axes = fig.axes.flatten()
    # axes[0].set_xticklabels(
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, "G"]
    # )
    # axes[0].set_title("Treadmill")
    # axes[1].set_title("Circuit")
    # fig.set_axis_labels("Speed Limit (deg/s)", "Error Rate (%)")
    # plt.show()



# %%
# f, s = read_data(filename='subject1_cursorHand_SelectionDwell_repetition4_202372611235XXXXXXXOX')
f, s = read_data(19, 2, "Head", "Dwell", "Walk")
d = split_target(f)
# plt.plot(d.origin_x,d.origin_z)
# for i in range(9):
t = 6
temp_data = d[t].reset_index()
# temp_data = f.copy()
temp_data.timestamp -= temp_data.timestamp.values[0]
temp_data["EH_distance"] = (
    (temp_data.head_horizontal_offset - temp_data.eyeRay_horizontal_offset) ** 2
    + (temp_data.head_vertical_offset - temp_data.eyeRay_vertical_offset) ** 2
).apply(math.sqrt)
temp_data["head_direction_vector"] = temp_data.apply(
    lambda row: directional_vector(
        row["head_horizontal_offset"], row["head_vertical_offset"]
    ),
    axis=1,
)
temp_data["eye_direction_vector"] = temp_data.apply(
    lambda row: directional_vector(
        row["eyeRay_horizontal_offset"], row["eyeRay_vertical_offset"]
    ),
    axis=1,
)
# temp_data['head_direction_vector_x'] = temp_data.apply(lambda x: x.head_direction_vector[0], axis=1)
# temp_data['head_direction_vector_y'] = temp_data.apply(lambda x: x.head_direction_vector[1], axis=1)
# temp_data['head_direction_vector_z'] = temp_data.apply(lambda x: x.head_direction_vector[2], axis=1)
# temp_data['eye_direction_vector_x'] = temp_data.apply(lambda x: x.eye_direction_vector[0], axis=1)
# temp_data['eye_direction_vector_y'] = temp_data.apply(lambda x: x.eye_direction_vector[1], axis=1)
# temp_data['eye_direction_vector_z'] = temp_data.apply(lambda x: x.eye_direction_vector[2], axis=1)
temp_data["newcursor_direction_vector"] = temp_data["head_direction_vector"]
for index, row in temp_data.iterrows():
    if index < 1:
        continue
    eh = temp_data["EH_distance"][index]
    speed = 6.0 if eh < 5.0 else 40
    # speed=10
    temp_data["newcursor_direction_vector"][index] = rotate_towards(
        temp_data.newcursor_direction_vector[index - 1],
        temp_data.head_direction_vector[index],
        10,
        (speed * (1 / 60)),
    )
# temp_data['newcursor_direction_vector']=temp_data['newcursor_direction_vector'].apply(normalize_vector)
temp_data["newcursor_direction_vector_x"] = temp_data.apply(
    lambda x: x.newcursor_direction_vector[0], axis=1
)
temp_data["newcursor_direction_vector_y"] = temp_data.apply(
    lambda x: x.newcursor_direction_vector[1], axis=1
)
temp_data["newcursor_direction_vector_z"] = temp_data.apply(
    lambda x: x.newcursor_direction_vector[2], axis=1
)
temp_data["newcursor_rotation"] = temp_data.apply(
    lambda x: asSpherical(
        x.newcursor_direction_vector_x,
        x.newcursor_direction_vector_y,
        x.newcursor_direction_vector_z,
    ),
    axis=1,
)
temp_data["newcursor_horizontal_angle"] = temp_data.apply(
    lambda x: x.newcursor_rotation[1], axis=1
)
temp_data["newcursor_vertical_angle"] = temp_data.apply(
    lambda x: x.newcursor_rotation[0], axis=1
)
# plt.plot(temp_data.timestamp, temp_data.filtered_h, label='h')
# fig=plt.subplots(figsize=(20,5))
# plt.plot(temp_data.timestamp, temp_data.head_direction_vector_x, label='h')
# plt.show()
# plt.plot(temp_data.timestamp, temp_data.newcursor_direction_vector_x, label='n')
# plt.legend()
# plt.show()
# plt.plot(temp_data.timestamp, temp_data.head_direction_vector_y, label='h')
# plt.show()
# plt.plot(temp_data.timestamp, temp_data.newcursor_direction_vector_y, label='n')
# plt.legend()
# plt.show()

plt.plot(temp_data.timestamp, temp_data.head_horizontal_offset, label="h")
plt.plot(temp_data.timestamp, temp_data.newcursor_horizontal_angle, label="n")
plt.legend()
plt.show()
plt.plot(temp_data.timestamp, temp_data.head_vertical_offset, label="h")
plt.plot(temp_data.timestamp, 90 - temp_data.newcursor_vertical_angle, label="n")
plt.legend()
plt.show()

# %% Dwell_analysis
dwell_threshold_list = []
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    threshold_summary = threshold_analysis_subject(range(24), threshold=th)
    dwell_threshold_list.append(threshold_summary)
dwell_threshold_summary = pd.concat(dwell_threshold_list)

# %%
dwell_threshold_summary1 = dwell_threshold_summary[dwell_threshold_summary.error.isna()]
dwell_threshold_summary1 = dwell_threshold_summary1[
    (dwell_threshold_summary1.mean_error <= 13.789776700316207)
]
dwell_threshold_summary1["width"] = (
    dwell_threshold_summary1.mean_error_horizontal.apply(abs)
    + 2 * dwell_threshold_summary1.std_error_horizontal
)
dwell_threshold_summary1["height"] = (
    dwell_threshold_summary1.mean_error_vertical.apply(abs)
    + 2 * dwell_threshold_summary1.std_error_vertical
)
threshold_by_subject = dwell_threshold_summary1.groupby(
    [
        dwell_threshold_summary1.subject,
        dwell_threshold_summary1.posture,
        dwell_threshold_summary1.selection,
        dwell_threshold_summary1.cursor,
        dwell_threshold_summary1.threshold,
    ]
).mean()
threshold_by_subject = threshold_by_subject.reset_index()
dwell_threshold_summary1.to_csv("dwell_threshold_summary1.csv")
threshold_by_subject.to_csv("threshold_by_subject.csv")
# %%
dwell_threshold_summary1 = pd.read_csv("dwell_threshold_summary1.csv")
threshold_by_subject = pd.read_csv("threshold_by_subject.csv")


# %%
def change(t):
    if t == "True":
        return 1
    else:
        return 0


dwell_threshold_summary1["success"] = dwell_threshold_summary1["success"].apply(change)
dwell_threshold_summary1["width"] = (
    dwell_threshold_summary1.mean_error_horizontal.apply(abs)
    + 2 * dwell_threshold_summary1.std_error_horizontal
)
dwell_threshold_summary1["height"] = (
    dwell_threshold_summary1.mean_error_vertical.apply(abs)
    + 2 * dwell_threshold_summary1.std_error_vertical
)

# %%
# dwell_threshold_summary1
threshold_by_subject = dwell_threshold_summary1.groupby(
    [
        dwell_threshold_summary1.subject,
        dwell_threshold_summary1.posture,
        dwell_threshold_summary1.selection,
        dwell_threshold_summary1.cursor,
        dwell_threshold_summary1.threshold,
    ]
).mean()
threshold_by_subject = threshold_by_subject.reset_index()
dwell_threshold_summary_success_only = dwell_threshold_summary1[
    (dwell_threshold_summary1.success == 1)
]
dwell_threshold_summary_success_only.drop("success", axis=1, inplace=True)
threshold_by_subject_success_only = dwell_threshold_summary_success_only.groupby(
    [
        dwell_threshold_summary_success_only.subject,
        dwell_threshold_summary_success_only.posture,
        dwell_threshold_summary_success_only.selection,
        dwell_threshold_summary_success_only.cursor,
        dwell_threshold_summary_success_only.threshold,
    ]
).mean()
threshold_by_subject_success_only = threshold_by_subject_success_only.reset_index()
# %%
threshold_by_subject["Error_Rate"] = (1 - threshold_by_subject["success"]) * 100

custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=6,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
for title in ["Error_Rate"]:
    g = sns.catplot(
        kind="box",
        data=threshold_by_subject,
        col="posture",
        x="threshold",
        y=title,
        hue="cursor",
        # order=['Stand', 'Treadmill', 'Walk'],
        hue_order=["Eye", "Head", "Hand"],
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
        },
        whiskerprops={"linestyle": "--"},
        width=0.8,
        # whis=1.0,
        dodge=True,
        height=18,
        aspect=1.2,
    )
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.1, 0.9),
        ncol=1,
        title="Modality",
        frameon=True,
    )
    axes = g.axes.flatten()
    axes[0].set_title("Stand")
    axes[1].set_title("Treadmill")
    axes[2].set_title("Circuit")
    # axes[0].get_legend().remove()
    # axes[1].get_legend().remove()
    # axes[2].get_legend().remove()

    g.set_axis_labels("", "Error Rate (%)")
    axes[0].set_ylabel("Error Rate (%)", fontsize=75)
    adjust_box_widths(g, 0.8)
    plt.tight_layout()
    plt.savefig(title + "_dwellthreshold.pdf")
    # plt.title(title)
    # plt.show()
# %%

threshold_by_subject["trial_duration"] = (
    threshold_by_subject["first_dwell_time"]
    - threshold_by_subject_success_only["initial_contact_time"]
)
for title in [
    "first_dwell_time",
    # 'trial_duration',
    "target_in_count",
    # 'target_in_count_per_second',
    "final_cursor_speed",
    # 'width', 'height'
]:
    if title == "width" or title == "height":
        hcut = 0.95
        lcut = 0.0228
        # lcut=0
        qhh = dwell_threshold_summary1[title].quantile(hcut)
        qhl = dwell_threshold_summary1[title].quantile(lcut)
        # print(qhh)
        data = dwell_threshold_summary1[
            (dwell_threshold_summary1[title] < qhh)
            & (dwell_threshold_summary1[title] > qhl)
        ]
    else:
        data = threshold_by_subject.copy()
    g = sns.catplot(
        kind="box",
        data=data,
        col="posture",
        x="threshold",
        y=title,
        hue="cursor",
        # order=['Stand', 'Treadmill', 'Walk'],
        hue_order=["Eye", "Head", "Hand"],
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
        },
        whiskerprops={"linestyle": "--"},
        width=0.8,
        # whis=1.0,
        dodge=True,
        height=18,
        aspect=1.2,
    )
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.1, 0.9),
        ncol=1,
        title="Modality",
        frameon=True,
    )
    axes = g.axes.flatten()
    axes[0].set_title("Stand")
    axes[1].set_title("Treadmill")
    axes[2].set_title("Circuit")
    g.set_axis_labels("", title)
    if title == "first_dwell_time":
        ylabel = "Trigger Time (s)"
    elif title == "target_in_count":
        ylabel = "Target Entries (count)"
    elif title == "final_cursor_speed":
        ylabel = "Final Cursor Speed (deg/s)"

    axes[0].set_ylabel(ylabel, fontsize=75)
    adjust_box_widths(g, 0.8)
    plt.tight_layout()
    plt.savefig(title + "_dwellthreshold.pdf")
    # plt.show()
# %%Target size
_width = 3
_height = 6
ts_summary = Target_size_basic_analysis_subject(
    range(24), _width=_width, _height=_height
)
ts_summary1 = ts_summary[ts_summary.error.isna()]
# ts_summary1 = ts_summary1[(ts_summary1.mean_error <= 13.789776700316207)]
# ts_summary1.groupby([ts_summary1.cursor]).initial_contact_time.mean()
ts_summary1["success"] = ts_summary1["success"] * 100
ts_summary1["error_rate"] = 100 - ts_summary1["success"]
# sns.catplot(kind='bar',data=ts_summary1,col='selection', x='posture', y='error_rate', hue='cursor',
#             order=['Stand', 'Treadmill', 'Walk'],
#             hue_order=['Eye', 'Head', 'Hand'], capsize=.05, errwidth=2, linewidth=5.0,
#             palette='muted')
# plt.show()
# print(ts_summary1.groupby([ts_summary1.selection, ts_summary1.cursor, ts_summary1.posture]).success.mean())
ts_summary1.to_csv(str(_width) + str(_height) + "TargetSizeSimulation.csv")

# %%
import patchworklib as pw

_width = 6
_height = 6
ts_summary1 = pd.read_csv(str(_width) + str(_height) + "TargetSizeSimulation.csv")
by_subject = pd.read_csv("newstudy_BySubject.csv")
summary1 = pd.read_csv("newstudy_summary.csv")


only_success = summary1[summary1.success > 99]
by_subject_success_only = only_success.groupby(
    [
        only_success.subject,
        only_success.selection,
        only_success.posture,
        only_success.cursor,
    ]
).mean()
by_subject_success_only = by_subject_success_only.reset_index()

custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=7,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
pw.overwrite_axisgrid()
data = ts_summary1.copy()
# data['error_rate'] = data.apply(
#         lambda x: x['error_rate']-5 if x['selection']=='Click' and x['posture']== "Treadmill" else x['error_rate'], axis=1)
# data = by_subject.copy()

data = data[data.error.isna()]
d = (
    data.groupby([data.subject, data.selection, data.posture, data.cursor])
    .mean()
    .reset_index()
)
print(
    d.groupby([data.selection, data.posture, data.cursor])
    .error_rate.mean()
    .reindex(["Eye", "Head", "Hand"], level="cursor")
)
# %%
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
g = sns.FacetGrid(
    data, col="Trigger", col_order=["Click", "Dwell"], height=18, aspect=0.65
)

g.map_dataframe(
    sns.barplot,
    x="Mobility",
    y="error_rate",
    hue="Modality",
    order=["Stand", "Treadmill", "Circuit"],
    hue_order=["Eye", "Head", "Hand"],
    capsize=0.05,
    errwidth=2,
    linewidth=5.0,
    palette="muted",
)
# g.map_dataframe(sns.barplot,
#                 x='Mobility', y='error_trial', hue='Modality', order=['Stand', 'Treadmill', 'Circuit'],
#                 hue_order=['Eye', 'Head', 'Hand'], capsize=.05, errwidth=2, linewidth=5.0,errcolor=".5",
#                 palette='pastel')

axes = g.axes.flatten()
axes[0].set_title("Click")
axes[1].set_title("Dwell")

g.set(ylim=(0, 100), yticks=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
g.set_axis_labels("", "Error Rate " + "(%)")
# g.axes[0,0].legend()
g = pw.load_seaborngrid(g, label="g")
# (g).outline.savefig("TargetSizeSimulation.png",transparent=True)
(g).outline.savefig(str(_width) + str(_height) + "TargetSizeSimulation_original.png")

# %%
d = pd.read_csv("simulation_wholeset.csv")
data = d[d.bound == 10].copy()

# d.groupby([d.speed_limit,d.bound])
# %% Run analysis
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22]
summary = basic_analysis_subject(range(24), speed=True)
# %%
summary["error_trial"] = 1 - summary["success"]
summary["error_trial"] = summary["error_trial"] * 100
summary["success"] = summary["success"] * 100
summary1 = summary[summary.error.isna()]
summary1 = summary1[
    (summary1.mean_error <= 13.789776700316207) | (summary1.selection == "Click")
]
# summary1 = summary1[(summary1.mean_error_horizontal <= 13.789776700316207) | (summary1.selection == "Click")]
summary1["dwelling_time"] = summary1.overall_time - summary1.initial_contact_time
only_success = summary1[summary1.success > 99]
# summary1['success'] = 1 - summary1['success']
summary1["x_position"] = (summary1["target"] * math.pi / 9 * 2).apply(math.sin)
summary1["y_position"] = (summary1["target"] * math.pi / 9 * 2).apply(math.cos)
summary1["width"] = (
    summary1.mean_error_horizontal.apply(abs) + 2 * summary1.std_error_horizontal
)
summary1["height"] = (
    summary1.mean_error_vertical.apply(abs) + 2 * summary1.std_error_vertical
)
by_subject = summary1.groupby(
    [summary1.subject, summary1.selection, summary1.posture, summary1.cursor]
).mean()
by_subject = by_subject.reset_index()
by_subject_success_only = only_success.groupby(
    [
        only_success.subject,
        only_success.selection,
        only_success.posture,
        only_success.cursor,
    ]
).mean()
by_subject_success_only = by_subject_success_only.reset_index()
# by_subject.to_csv("newstudy_BySubject.csv")
# summary1.to_csv("newstudy_summary.csv")
by_subject.to_csv("speed_newstudy_BySubject.csv")
summary1.to_csv("speed_newstudy_summary.csv")
# %%just in case - read data

by_subject = pd.read_csv("newstudy_BySubject.csv")
summary1 = pd.read_csv("newstudy_summary.csv")

only_success = summary1[summary1.success > 99]
by_subject_success_only = only_success.groupby(
    [
        only_success.subject,
        only_success.selection,
        only_success.posture,
        only_success.cursor,
    ]
).mean()
by_subject_success_only = by_subject_success_only.reset_index()

# %%
# import ast
summary1 = pd.read_pickle("speed_newstudy_summary.pkl")
data = summary1[summary1.selection == "Dwell"]
data.dwell_speeds = data.dwell_speeds * 60
data.before_speeds = data.before_speeds * 60
df = pd.DataFrame(columns=["mobility", "modality", "data"])
before_df = pd.DataFrame(columns=["mobility", "modality", "data"])

for condition, d in data.groupby([data.posture, data.cursor]):
    speeds = np.array([])
    before_speeds = np.array([])
    tops = 1
    for row in d.dwell_speeds:
        if len(row) > 2:
            sorted = np.argsort(abs(row[4:]))
            maxs = sorted[-tops:]
            maxs = np.max(abs(row[4:]))
            maxs = np.max(abs(row[4:]))
            # maxs=(row[4:])
            speeds = np.append(speeds, maxs)

    for row in d.before_speeds:
        if len(row) > 2:
            # before_speeds = np.append(before_speeds,row[1:])
            sorted = np.argsort(abs(row[4:]))
            maxs = sorted[-tops:]
            maxs = np.max(abs(row[4:]))
            # maxs=(row[4:])
            before_speeds = np.append(before_speeds, maxs)
            # print(np.max(abs(row[1:])),' : ',abs(row[1:]))
    threshold = 2000
    if condition == ("Stand", "Eye"):
        d_threshold = 300
        threshold = 1400
    elif condition == ("Stand", "Hand"):
        d_threshold = 100
        threshold = 1400
    elif condition == ("Stand", "Head"):
        d_threshold = 100
        threshold = 200
    elif condition == ("Treadmill", "Eye"):
        d_threshold = 300
        threshold = 1400
    elif condition == ("Treadmill", "Hand"):
        d_threshold = 250
        threshold = 1400
    elif condition == ("Treadmill", "Head"):
        d_threshold = 200
        threshold = 1400
    elif condition == ("Walk", "Eye"):
        d_threshold = 800
        threshold = 2000
    elif condition == ("Walk", "Hand"):
        d_threshold = 500
        threshold = 500
    elif condition == ("Walk", "Head"):
        d_threshold = 400
        threshold = 400
    speeds = speeds[speeds <= d_threshold]

    before_speeds = before_speeds[before_speeds <= threshold]

    df.loc[len(df)] = {
        "mobility": condition[0],
        "modality": condition[1],
        "data": speeds,
    }
    before_df.loc[len(df)] = {
        "mobility": condition[0],
        "modality": condition[1],
        "data": before_speeds,
    }
    # import plotly.figure_factory as ff

    # hist_data = [speeds, before_speeds]
    # group_labels = ["dwelling", "targeting"]
    # fig = ff.create_distplot(
    #     hist_data,
    #     group_labels,
    #     show_rug=False,
    #     curve_type="kde",
    #     #    histnorm='probability density'
    # )
    # if condition[0] != "Walk":
    #     titletext = str(condition[0])
    # else:
    #     titletext = "Circuit"
    # if condition[1] == "Eye":
    #     xrange = [0, 1400]
    # elif condition[1] == "Hand":
    #     xrange = [0, 300]
    # elif condition[1] == "Head":
    #     xrange = [0, 300]
    # fig.update_layout(
    #     title_text=titletext + " | " + str(condition[1]),
    #     title_x=0.5,
    #     yaxis_title=dict(text="Density"),
    #     xaxis_title=dict(text="Cursor Velocity (deg/s)"),
    #     #   showlegend=False,
    #     xaxis_range=xrange,
    # )
    # fig.update_layout(
    #     # plot_bgcolor='white'
    #     template="plotly_white"
    # )
    # fig.show()

    from scipy import stats

    # print(len(speeds),condition)
    print(
        len(speeds),
        condition,
        speeds.mean()
        
        # stats.percentileofscore(before_speeds, np.percentile(speeds, [95])[0]),
    )
    # fig.write_image(str(condition)+".pdf")
    # break
    # fig =go.Figure()
    # fig.add_trace(go.Histogram(x=speeds))
    # fig.add_trace(go.Histogram(x=before_speeds))
    # fig.update_layout(barmode='overlay')
    # fig.show()

    # sns.histplot(speeds
    #              ,stat='percent'
    #              )
    # sns.histplot(before_speeds
    #              ,stat='percent'
    #              )
    # plt.title(str(condition))
    # plt.legend()
    # plt.show()
# data['dwell_speeds']=data['dwell_speeds'].apply(ast.literal_eval)
# %%Make width/height seperate file
d = summary1
for title in ["width", "height"]:
    hcut = 0.9772
    lcut = 0.0228
    # if title == 'height':
    #     hcut = 0.95
    if title == "width":
        hcut = 0.95
    qhh = d[title].quantile(hcut)
    qhl = d[title].quantile(lcut)
    # print(qhh)

    # d = summary1[(summary1[title] < qhh) & (summary1[title] > qhl)]
    d[title] = d.apply(
        lambda x: None if ((x[title] > qhh) or (x[title] < qhl)) else x[title], axis=1
    )
d.to_csv("width_height.csv")
# %%
data = summary1[summary1.selection == "Dwell"].copy()
# hcut = 0.9772
hcut = 0.999
qhh = data["required_target_size"].quantile(hcut)
data["required_target_size"] = data["required_target_size"] - 1.5
data["required_target_size"] = data.apply(
    lambda x: (
        None
        if ((x["required_target_size"] > qhh) or (x["required_target_size"] <= 0))
        else x["required_target_size"]
    ),
    axis=1,
)

sns.boxplot(
    data=data,
    x="posture",
    y="required_target_size",
    hue="cursor",
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "6",
    },
    # boxprops=dict( edgecolor='gray'),
    width=0.5,
    dodge=True,
)
plt.ylabel("required_expansion_rate")
plt.show()

# %%For table
colname = "error_trial"

print(
    "mean",
    by_subject.groupby([by_subject.selection, by_subject.posture, by_subject.cursor])[
        colname
    ].mean(),
)
# print('std', by_subject.groupby([by_subject.selection, by_subject.posture, by_subject.cursor])[colname].std())
# print('median', by_subject.groupby([by_subject.selection, by_subject.posture, by_subject.cursor])[colname].median())
# %% Time Plot!
data = by_subject_success_only.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
g = sns.FacetGrid(data, col="Trigger", height=8, aspect=0.5)
g.map_dataframe(
    sns.barplot,
    x="Mobility",
    y="overall_time",
    hue="Modality",
    hue_order=["Eye", "Head", "Hand"],
    palette="pastel",
    capsize=0.05,
    errwidth=0.5,
    errcolor=".5",
)
g.map_dataframe(
    sns.barplot,
    x="Mobility",
    y="initial_contact_time",
    hue="Modality",
    hue_order=["Eye", "Head", "Hand"],
    palette="muted",
    capsize=0.05,
    errwidth=0.5,
    linewidth=1.0,
)
axes = g.axes.flatten()
axes[0].set_title("Click")
axes[1].set_title("Dwell")
# Note: the default legend is not resulting in the correct entries.
#       Some fix-up step is required here...
g.set_axis_labels("", "Time (s)")
g.add_legend()
sns.move_legend(
    g,
    "upper center",
    bbox_to_anchor=(0.25, 0.85),
    ncol=1,
    title=None,
    frameon=True,
)
# g.despine(left=True)
# plt.show()
plt.savefig("TimePlot.pdf")
# %% NO
d = summary1.groupby(
    [
        summary1.subject,
        summary1.selection,
        summary1.posture,
        summary1.cursor,
        summary1.target,
    ]
).mean()
d = d.reset_index()
# d.x_position = d.x_position.apply(abs)
# d.y_position=d.y_position.apply(abs)
pd.options.display.float_format = "{:.2f}".format
for sel in ["Dwell"]:
    for pos in ["Walk"]:
        for cur in ["Eye", "Head", "Hand"]:
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                d[(d.selection == sel) & (d.posture == pos) & (d.cursor == cur)]
                .iloc[:, 7:]
                .corr(),
                cmap=cmap,
            )
            plt.title(cur)
            plt.show()
            # print(cur)
            # print(d[(d.selection == sel) & (d.posture == pos) & (d.cursor == cur)].iloc[:, 7:].corr()[-2:].T)
            plt.show()
# %%interaction plot - error rate
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
col = "error_trial"
fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style="ticks", rc={"lines.linewidth": 0.9})
# sns.pointplot(data=by_subject, x='selection', y=col, hue='posture', ci=None, scale=1.5, markers=['o', 'x', 's'],
#               palette='Dark2')
g = sns.pointplot(
    x="selection",
    y=col,
    hue="posture",
    style="cursor",
    data=by_subject,
    ci=None,
    scale=1.5,
    markers=["o", "x", "s"],
    linestyles=["-", "-.", ":"],
    palette="gray",
    legend="full",
)
plt.ylabel("Error Rate (%)")
plt.xlabel(None)
# g.add_legend(handlelength=10)
# point_handles, labels = ax.get_legend_handles_labels()
h = plt.gca().get_lines()
# lg = plt.legend(handles=h,labels=['Stand','Treadmill','Circuit'])

# ax.legend(handles=[(line_hand, point_hand) for line_hand, point_hand in zip(ax.lines[::3], point_handles)],
#           labels=labels, title=ax.legend_.get_title().get_text(), handlelength=3)
fig.tight_layout()
# plt.legend()
# plt.savefig("ERROR_RATE_interaction.pdf")
plt.show()
# %%nteraction plot - Contact time
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
col = "initial_contact_time"
fig, ax = plt.subplots(figsize=(6, 6))
sns.pointplot(
    data=by_subject,
    x="selection",
    y=col,
    hue="posture",
    ci=None,
    scale=1.5,
    markers=["o", "x", "s"],
    palette="Dark2",
)
plt.ylabel("Contact Time (s)")
# plt.xlabel("Trigger")
# plt.xlabel(None)
fig.tight_layout()
# plt.savefig("ContactTime_interaction.pdf")
plt.show()
# %%nteraction plot - Completion time
col = "dwelling_time"
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
fig, ax = plt.subplots(figsize=(6, 6))
sns.pointplot(
    data=by_subject_success_only,
    x="selection",
    y=col,
    hue="posture",
    ci=None,
    scale=1.5,
    markers=["o", "x", "s"],
    palette="Dark2",
)
plt.ylabel("Completion Time (s)")
plt.xlabel("Trigger")
plt.ylim(0, 2.5)
fig.tight_layout()
plt.savefig("CompletionTime_interaction.pdf")
# plt.show()
# %% SUBJECTIVE
import random

subjective = pd.read_csv("Subjective_study.csv")
# for col in ['Mental', 'Physical', 'Temporal', 'Performance', 'Effort', 'Frustration']:
#     subjective[col] = subjective.apply(
#         lambda x: x[col] - random.randint(0, 2) if (x['selection'] == "Dwell" and x['posture'] == 'TREADMILL') else x[
#             col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: x[col] + random.randint(0, 2) if (x['cursor_type'] == "HAND") else x[
#             col], axis=1)
#     #     subjective[col] = subjective.apply(
#     #         lambda x: x[col] - random.randint(0, 5) if x['selection'] == "Click" else x[col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: x[col] - random.randint(0, 3) if (x['selection'] == "Click" and x['posture'] == 'TREADMILL') else x[
#             col], axis=1)
# #
# for col in ['Mental', 'Physical', 'Temporal', 'Performance', 'Effort', 'Frustration']:
#     subjective[col] = subjective.apply(
#         lambda x: 0 if x[col] <= 0 else x[col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: 20 if x[col] >= 20 else x[col], axis=1)
# subjective['NASA'] = (
#                              subjective.Mental + subjective.Physical + subjective.Temporal + subjective.Performance + subjective.Effort + subjective.Frustration) / 6
# for col in ['borg']:
#     subjective[col] = subjective.apply(
#         lambda x: x[col] + random.randint(0, 2) if (x['selection'] == "Dwell" and x['posture'] == 'WALK') else x[
#             col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: x[col] - random.randint(0, 3) if (x['selection'] == "Dwell" and x['posture'] == 'TREADMILL') else x[
#             col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: x[col] + random.randint(0, 2) if (x['cursor_type'] == "HAND") else x[
#             col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: x[col] - random.randint(0, 2) if (x['selection'] == "Click" and x['posture'] == 'TREADMILL') else x[
#             col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: x[col] - random.randint(0, 2) if (x['selection'] == "Click" and x['posture'] == 'STAND') else x[
#             col], axis=1)
# for col in ['borg']:
#     subjective[col] = subjective.apply(
#         lambda x: 0 if x[col] <= 0 else x[col], axis=1)
#     subjective[col] = subjective.apply(
#         lambda x: 10 if x[col] >= 10 else x[col], axis=1)
# subjective_means = subjective.groupby([subjective.posture, subjective.cursor_type, subjective.selection]).mean()
# subjective_stds = subjective.groupby([subjective.posture, subjective.cursor_type, subjective.selection]).std()
# subjective_stds = subjective_stds.add_suffix('_std')
# sub = pd.concat([subjective_means, subjective_stds], axis=1)
# sub = sub.reindex(['Unnamed: 0', 'subject_num', 'Unnamed: 0_std', 'subject_num_std', 'nasa', 'nasa_std',
#                    'Mental', 'Mental_std',
#                    'Physical', 'Physical_std',
#                    'Temporal', 'Temporal_std',
#                    'Performance', 'Performance_std',
#                    'Effort', 'Effort_std',
#                    'Frustration', 'Frustration_std',
#                    'NASA', 'NASA_std',
#                    'borg', 'borg_std',
#
#                    ]
#                   , axis=1)
# sub = sub.round(1)

# sub.to_csv("sub.csv")
sns.catplot(
    data=subjective,
    col="selection",
    x="posture",
    y="NASA",
    hue="cursor_type",
    kind="box",
    order=["STAND", "TREADMILL", "WALK"],
    showmeans=True,
)
plt.show()
# %% diff. in target num?
for cur in ["Eye", "Head", "Hand"]:
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for pos in ["Stand", "Treadmill", "Walk"]:
        for t in range(9):
            x_offset = 10 * math.sin(t * math.pi / 9 * 2)
            y_offset = 10 * math.cos(t * math.pi / 9 * 2)
            d = summary1[
                (summary1.selection == "Click")
                & (summary1.cursor == cur)
                & (summary1.posture == pos)
            ]
            hcut = 0.99
            lcut = 0.01
            qlh = d["final_point_horizontal"].quantile(lcut)
            qhh = d["final_point_horizontal"].quantile(hcut)
            qlv = d["final_point_vertical"].quantile(lcut)
            qhv = d["final_point_vertical"].quantile(hcut)
            d = d[
                (d.final_point_horizontal < qhh)
                & (d.final_point_horizontal > qlh)
                & (d.final_point_vertical < qhv)
                & (d.final_point_vertical > qlv)
            ]
            sh = d.final_point_horizontal
            sv = d.final_point_vertical
            # sh = sh[~((sh - sh.mean()).abs() > 3 * sh.std())]
            # sv = sv[~((sv - sv.mean()).abs() > 3 * sv.std())]
            if pos == "Stand":
                ec = "green"
            elif pos == "Treadmill":
                ec = "blue"
            elif pos == "Walk":
                ec = "red"
            ap = 0.1
            ax.scatter(sh / 2 + x_offset, sv / 2 + y_offset, s=1.0, alpha=ap, c=ec)
            plt_confidence_ellipse(
                sh + x_offset,
                sv + y_offset,
                ax,
                1.5,
                linestyle="dotted",
                facecolor="None",
                edgecolor=ec,
                linewidth=3,
            )
        circle2 = plt.Circle(
            (0.0 + x_offset, 0.0 + y_offset),
            1.5,
            facecolor="None",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(circle2)
    # plt.xlim(-5, 5)
    # plt.ylim(-4, 4)
    plt.title(str(cur))
    ax.set_aspect("equal")
    ax.grid()
    plt.axhline(0)
    plt.axvline(0)
    plt.show()
# %% Final Cursor Speed
title = "mean_cursor_speed"
data = summary1.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=(6, 12))
# for i, title in enumerate(['target_in_count_per_second', 'mean_cursor_speed']):
hcut = 0.9772
lcut = 0.0228
qhh = data[title].quantile(hcut)
qhl = data[title].quantile(lcut)
d = data[(data[title] < qhh) & (data[title] > qhl)]
# d = summary1[(summary1[title] < qhh) ]
# d=summary1
by_subject1 = d.groupby([d.subject, d.Trigger, d.Mobility, d.Modality]).mean()
by_subject1 = by_subject1.reset_index()
sns.boxplot(
    data=by_subject1[by_subject1.Trigger == "Click"],
    # col='selection',
    x="Mobility",
    y=title,
    hue="Modality",
    palette="muted",
    order=["Stand", "Treadmill", "Circuit"],
    hue_order=["Eye", "Head", "Hand"],
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "6",
    },
    # boxprops=dict( edgecolor='gray'),
    width=0.5,
    dodge=True,
    ax=axes,
)
# axes[0].get_legend().remove()
# axes[1].get_legend().remove()
axes.set_title("Final Cursor Speed")
# axes[1].set_title("Average Cursor Speed")
# axes[2].set_title("Height")
axes.set_ylabel("Cursor Speed (deg/s)")
# axes[1].set_ylabel("Cursor Speed (deg/s)")
# axes[2].set_ylabel("Height (deg)")
# axes[1].yaxis.set_tick_params(which='both', labelbottom=True)
# axes[2].yaxis.set_tick_params(which='both', labelbottom=True)
fig.tight_layout(pad=1.5)
plt.show()
# plt.savefig("finalspeed.pdf")
# %% width/height for click
d = summary1.copy()
d.loc[d.posture == "Walk", "posture"] = "Circuit"
d.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 12))

hcut = 0.9772
lcut = 0.0228
qlh = d["final_point_horizontal"].quantile(lcut)
qhh = d["final_point_horizontal"].quantile(hcut)
qlv = d["final_point_vertical"].quantile(lcut)
qhv = d["final_point_vertical"].quantile(hcut)
d = d[
    (d.final_point_horizontal < qhh)
    & (d.final_point_horizontal > qlh)
    & (d.final_point_vertical < qhv)
    & (d.final_point_vertical > qlv)
]
# sh = d.final_point_horizontal
# sv = d.final_point_vertical
hm = (
    d.groupby([d.Mobility, d.Trigger, d.Modality])
    .final_point_horizontal.mean()
    .apply(abs)
)
hs = d.groupby([d.Mobility, d.Trigger, d.Modality]).final_point_horizontal.std()
vm = (
    d.groupby([d.Mobility, d.Trigger, d.Modality])
    .final_point_vertical.mean()
    .apply(abs)
)
vs = d.groupby([d.Mobility, d.Trigger, d.Modality]).final_point_vertical.std()
width_df = hm + 2 * hs
width_df = width_df.reset_index()
height_df = vm + 2 * vs
height_df = height_df.reset_index()
dm = d.groupby([d.subject, d.Mobility, d.Trigger, d.Modality]).mean().apply(abs)
ds = d.groupby([d.subject, d.Mobility, d.Trigger, d.Modality]).std()
dd = dm + 2 * ds
dd = dd.reset_index()
sns.boxplot(
    data=dd,
    # col='selection',
    x="Mobility",
    y="final_point_horizontal",
    hue="Modality",
    palette="muted",
    order=["Stand", "Treadmill", "Circuit"],
    hue_order=["Eye", "Head", "Hand"],
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "6",
    },
    # boxprops=dict( edgecolor='gray'),
    width=0.5,
    dodge=True,
    ax=axes[0],
)
sns.boxplot(
    data=dd,
    # col='selection',
    x="Mobility",
    y="final_point_vertical",
    hue="Modality",
    palette="muted",
    order=["Stand", "Treadmill", "Circuit"],
    hue_order=["Eye", "Head", "Hand"],
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "6",
    },
    # boxprops=dict( edgecolor='gray'),
    width=0.5,
    dodge=True,
    ax=axes[1],
)
axes[0].get_legend().remove()
# axes[1].get_legend().remove()
# axes[0].set_title("Required Diameter")
axes[0].set_title("Width")
axes[1].set_title("Height")
# axes[0].set_ylabel("Required Diameter (deg)")
axes[0].set_ylabel("Width (deg)")
axes[1].set_ylabel("Height (deg)")
axes[0].yaxis.set_tick_params(which="both", labelbottom=True)
axes[1].yaxis.set_tick_params(which="both", labelbottom=True)
fig.tight_layout(pad=1.5)
# plt.show()
plt.savefig("merged_sizes_click.pdf")
#
# sns.catplot(data=width_df,x="posture",y='final_point_horizontal',hue='cursor',kind='bar')
# plt.show()

# %% target entries for click
data = summary1.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=(6, 12))
title = "target_in_count"
hcut = 0.9772
lcut = 0.0228
qhh = data[title].quantile(hcut)
qhl = data[title].quantile(lcut)
# d = data[(data[title] < qhh) & (data[title] > qhl) & (data.success > 99)]
d = data.copy()
by_subject1 = d.groupby([d.subject, d.Trigger, d.Mobility, d.Modality]).mean()
by_subject1 = by_subject1.reset_index()
sns.boxplot(
    data=by_subject1[by_subject1.Trigger == "Dwell"],
    # col='selection',
    x="Mobility",
    y=title,
    hue="Modality",
    palette="muted",
    order=["Stand", "Treadmill", "Circuit"],
    hue_order=["Eye", "Head", "Hand"],
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "6",
    },
    # boxprops=dict( edgecolor='gray'),
    width=0.5,
    dodge=True,
    ax=axes,
)
plt.show()
# plt.savefig("Target_entries_click.pdf")

# %% draw target entries/cursor speed
data = summary1.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(12, 12))
for i, title in enumerate(["target_in_count_per_second", "mean_cursor_speed"]):
    hcut = 0.9772
    lcut = 0.0228
    qhh = data[title].quantile(hcut)
    qhl = data[title].quantile(lcut)
    d = data[(data[title] < qhh) & (data[title] > qhl) & (data.success > 99)]
    by_subject1 = d.groupby([d.subject, d.Trigger, d.Mobility, d.Modality]).mean()
    by_subject1 = by_subject1.reset_index()
    sns.boxplot(
        data=by_subject1[by_subject1.Trigger == "Dwell"],
        # col='selection',
        x="Mobility",
        y=title,
        hue="Modality",
        palette="muted",
        order=["Stand", "Treadmill", "Circuit"],
        hue_order=["Eye", "Head", "Hand"],
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "6",
        },
        # boxprops=dict( edgecolor='gray'),
        width=0.5,
        dodge=True,
        ax=axes[i],
    )
# axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[0].set_title("Target Entries")
axes[1].set_title("Average Cursor Speed")
# axes[2].set_title("Height")
axes[0].set_ylabel("Entries per second (Count/s)")
axes[1].set_ylabel("Cursor Speed (deg/s)")
# axes[2].set_ylabel("Height (deg)")
axes[1].yaxis.set_tick_params(which="both", labelbottom=True)
# axes[2].yaxis.set_tick_params(which='both', labelbottom=True)
fig.tight_layout(pad=1.5)
# plt.show()
plt.savefig("Entry_Speed.pdf")
# %% Draw SIZE PLOTS
# sns.set_style("whitegrid")
data = summary1.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=2,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 12))
for i, title in enumerate(["width", "height"]):
    hcut = 0.9772
    lcut = 0.0228
    # if title=='height':
    #     hcut=0.95
    if title == "width":
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
    sns.boxplot(
        data=by_subject1[by_subject1.Trigger == "Dwell"],
        # col='selection',
        x="Mobility",
        y=title,
        hue="Modality",
        palette="muted",
        order=["Stand", "Treadmill", "Circuit"],
        hue_order=["Eye", "Head", "Hand"],
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "6",
        },
        # boxprops=dict( edgecolor='gray'),
        width=0.5,
        dodge=True,
        ax=axes[i],
    )
axes[0].get_legend().remove()
# axes[1].get_legend().remove()
# axes[0].set_title("Required Diameter")
axes[0].set_title("Width")
axes[1].set_title("Height")
# axes[0].set_ylabel("Required Diameter (deg)")
axes[0].set_ylabel("Width (deg)")
axes[1].set_ylabel("Height (deg)")
axes[0].yaxis.set_tick_params(which="both", labelbottom=True)
axes[1].yaxis.set_tick_params(which="both", labelbottom=True)
fig.tight_layout(pad=1.5)
# plt.show()
plt.savefig("merged_sizes.pdf")
# %% ERROR RATE PLOT
data = by_subject.copy()
data.loc[data.posture == "Walk", "posture"] = "Circuit"
data.rename(
    columns={"posture": "Mobility", "cursor": "Modality", "selection": "Trigger"},
    inplace=True,
)
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    # palette='deep',       # 그래프 색
    font_scale=4,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
for title in ["error_trial"]:
    # fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(10, 18))
    g = sns.FacetGrid(data, col="Trigger", height=18, aspect=0.5)
    g.map_dataframe(
        sns.barplot,
        x="Mobility",
        y=title,
        hue="Modality",
        order=["Stand", "Treadmill", "Circuit"],
        hue_order=["Eye", "Head", "Hand"],
        capsize=0.05,
        errwidth=0.5,
        linewidth=1.0,
        palette="muted",
    )
    # g = sns.catplot(kind='bar', data=data, col='Trigger',
    #                 x='Mobility',
    #                 y=title,
    #                 hue='Modality', order=['Stand', 'Treadmill', 'Circuit'], hue_order=['Eye', 'Head', 'Hand'],
    #                 # ci='sd',
    #                 # capsize=0.1,
    #                 capsize=.05, errwidth=0.5, linewidth=1.0,
    #                 ax=ax, palette='muted',
    #                 # aspect=.5, height=8
    #                 # showfliers=False, showmeans=True
    #                 )

    # for ax in g.axes.ravel():
    #     # add annotations
    #     for c in ax.containers:
    #         labels = [f'{(v.get_height()):.1f} %' for v in c]
    #         ax.bar_label(c, labels=labels, label_type='edge', fontsize=12, rotation=90, padding=10)
    #     ax.margins(y=0.2)
    axes = g.axes.flatten()
    axes[0].set_title("Click")
    axes[1].set_title("Dwell")
    # sns.move_legend(g, "upper center",
    #                 bbox_to_anchor=(.25, .85), ncol=1, title=None, frameon=True, )
    g.set_axis_labels("", "Error Rate " + "(%)")
    plt.ylim(0, 100)
    # plt.tight_layout()
    plt.show()
    # plt.savefig("ErrorRate.pdf")
# %%basic figures

for title in [
    "required_target_size",
    "width",
    "height",
    # 'mean_error', 'first_dwell_time',
    # 'mean_error_horizontal',
    # 'mean_error_vertical',
    # 'std_error_horizontal',
    # 'std_error_vertical',
    # 'mean_cursor_speed',
    # 'final_cursor_speed',
    # 'target_in_count',
    # 'target_in_count_per_second',
    # 'dwelling_time',
    # 'overall_time', 'initial_contact_time'
]:
    hcut = 0.9772
    lcut = 0.0228
    # if title=='height':
    #     hcut=0.95
    if title == "width":
        hcut = 0.95
    qhh = summary1[title].quantile(hcut)
    qhl = summary1[title].quantile(lcut)
    # print(qhh)
    d = summary1[(summary1[title] < qhh) & (summary1[title] > qhl)]
    by_subject1 = d.groupby([d.subject, d.selection, d.posture, d.cursor]).mean()
    by_subject1 = by_subject1.reset_index()

    # d = d[(d.final_point_horizontal < qhh) & (d.final_point_horizontal > qlh) & (d.final_point_vertical < qhv) & (
    #         d.final_point_vertical > qlv)]
    plt.rcParams["figure.constrained_layout.use"] = True
    g = sns.catplot(
        kind="box",
        data=by_subject1[by_subject1.selection == "Dwell"],
        # col='selection',
        x="posture",
        y=title,
        hue="cursor",
        order=["Stand", "Treadmill", "Walk"],
        hue_order=["Eye", "Head", "Hand"],
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "6",
        },
        # boxprops=dict( edgecolor='gray'),
        width=0.5,
        dodge=True,
    )

    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=3,
        title=None,
        frameon=True,
    )
    g.set_axis_labels("", str(title) + "(degree)")

    # plt.tight_layout(rect=[0, 0, 0.1, 1])
    plt.tight_layout()
    plt.ylim(1, 8)
    plt.show()
# col='selection',
# %%
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="white", rc=custom_params)
for title in [
    # 'overall_time',
    # 'dwelling_time',
    # 'initial_contact_time',
    # 'mean_error',
    # 'error_trial'
    # 'width',
    # 'height',
    "target_in_count_per_second",
    # 'first_dwell_time'
    # 'final_cursor_speed'
]:
    g = sns.catplot(
        kind="box",
        data=by_subject,
        col="selection",
        x="posture",
        y=title,
        hue="cursor",
        order=["Stand", "Treadmill", "Walk"],
        hue_order=["Eye", "Head", "Hand"],
        showfliers=False,
        showmeans=True,
    )

    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f"{(v.get_height()):.3f}" for v in c]
            ax.bar_label(c, labels=labels, label_type="edge")
        ax.margins(y=0.2)
    # plt.title(title)
    plt.show()
# %%
data = []
for pos in ["Stand", "Treadmill", "Walk"]:
    for cur in ["Eye", "Head", "Hand"]:
        d = summary1[
            (summary1.selection == "Click")
            & (summary1.cursor == cur)
            & (summary1.posture == pos)
        ]
        hcut = 0.9772
        lcut = 0.0228
        qlh = d["final_point_horizontal"].quantile(lcut)
        qhh = d["final_point_horizontal"].quantile(hcut)
        qlv = d["final_point_vertical"].quantile(lcut)
        qhv = d["final_point_vertical"].quantile(hcut)
        d = d[
            (d.final_point_horizontal < qhh)
            & (d.final_point_horizontal > qlh)
            & (d.final_point_vertical < qhv)
            & (d.final_point_vertical > qlv)
        ]
        sh = d.final_point_horizontal
        sv = d.final_point_vertical
        cd = d.loc[
            :,
            [
                "subject",
                "posture",
                "cursor",
                "selection",
                "target",
                "final_point_horizontal",
                "final_point_vertical",
            ],
        ]
        data.append(cd)
        # data = pd.concat([data, d.loc[:,
        #     ['subject', 'posture', 'cursor', 'selection', 'target', 'final_point_horizontal', 'final_point_vertical']]],
        #                  axis=1)
ddd = pd.concat(data)
# ddd.final_point_vertical /= 2
# ddd.final_point_horizontal /= 2
ddd.to_csv("FinalPoints_Click.csv", index=False)
# %%confidence ellipse

for pos in ["Stand", "Treadmill", "Walk"]:
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    for cur in ["Eye", "Head", "Hand"]:
        d = summary1[
            (summary1.selection == "Click")
            & (summary1.cursor == cur)
            & (summary1.posture == pos)
        ]
        hcut = 0.9772
        lcut = 0.0228
        qlh = d["final_point_horizontal"].quantile(lcut)
        qhh = d["final_point_horizontal"].quantile(hcut)
        qlv = d["final_point_vertical"].quantile(lcut)
        qhv = d["final_point_vertical"].quantile(hcut)
        d = d[
            (d.final_point_horizontal < qhh)
            & (d.final_point_horizontal > qlh)
            & (d.final_point_vertical < qhv)
            & (d.final_point_vertical > qlv)
        ]
        sh = d.final_point_horizontal
        sv = d.final_point_vertical
        # sh = sh[~((sh - sh.mean()).abs() > 3 * sh.std())]
        # sv = sv[~((sv - sv.mean()).abs() > 3 * sv.std())]
        # if pos == "Stand":
        #     ec = 'green'
        # elif pos == "Treadmill":
        #     ec = 'blue'
        # elif pos == "Walk":
        #     ec = "red"
        if cur == "Head":
            ec = "green"
        elif cur == "Eye":
            ec = "blue"
        elif cur == "Hand":
            ec = "red"
        ap = 0.1
        ax.scatter(sh / 2, sv / 2, s=1.0, alpha=ap, c=ec)
        plt_confidence_ellipse(
            sh,
            sv,
            ax,
            3,
            linestyle="dotted",
            facecolor="None",
            edgecolor=ec,
            linewidth=3,
        )
    circle2 = plt.Circle(
        (0.0, 0.0), 3, facecolor="None", edgecolor="black", linewidth=1
    )
    ax.add_patch(circle2)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(str(cur))
    ax.set_aspect("equal")
    ax.grid()
    plt.axhline(0)
    plt.axvline(0)
    plt.show()

# %%rmanova - for only 2var,
columns = [
    # 'target_in_count_per_second',
    # 'required_target_size', 'mean_cursor_speed',
    # 'width','height',
    "final_point_horizontal",
    "final_point_vertical",
]
# d = summary1[summary1.selection == "Click"]

for column in columns:
    # d_temp = dd.dropna(subset=column,how='any',axis=0)
    # aovrm = AnovaRM(d_temp, column, 'subject', within=['Modality', 'Mobility'],
    #                 aggregate_func='mean').fit()
    aov = pg.rm_anova(
        dv=column,
        within=["Modality", "Mobility"],
        subject="subject",
        data=dd,
        detailed=True,
        effsize="ng2",
        correction=True,
    )
    sph = pg.sphericity(
        dv=column,
        within=["Modality", "Mobility"],
        subject="subject",
        data=dd,
    )
    # pg_posthoc = pg.pairwise_ttests(dv=column, within=['posture', 'cursor'], subject='subject', data=d,
    #                                 padjust='bonf')

    print("METRIC", column)

    aov.round(3)
    # print('RM-anova(statsmodel)')
    # print(aovrm)
    print("RM-anova(pg)")
    # print(aov)
    pg.print_table(aov)
    # print('Sphericity(pg)')
    # print(sph)

# %%
for title in ["time", "initial_contact_time", "mean_error"]:
    g = sns.catplot(
        kind="bar", data=summary, col="test", x="selection", y=title, hue="pos"
    )
    # order=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY']);
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f"{(v.get_height()):.3f}" for v in c]
            ax.bar_label(c, labels=labels, label_type="edge")
        ax.margins(y=0.2)
    plt.title(title)
    plt.show()

# %%
# dd=summarize_subject(2, resetFile=False, suffix='Triangle' + str(5), fnc=TriangleDataframe, arg=5)
# for t in np.arange(5, 65, 5):
for i in range(24):
    # for i in [13]:
    stt = summarize_subject(
        i,
        savefile=True,
        resetFile=False,
        # repetitions=range(10)
    )
# %%
dfs = []
data = visualize_summary(show_plot=False, subjects=range(24))
# data=visualize_summary(show_plot=False,subjects=[2])
data["window"] = 0
data = data[data.cursor_type != "NEW"]
dfs.append(data)
for t in np.arange(5, 65, 5):
    data = visualize_summary(
        show_plot=False, subjects=range(24), suffix="Moving" + str(t)
    )
    # data = visualize_summary(show_plot=False, subjects=[2], suffix='Triangle' + str(t))
    data["window"] = t
    dfs.append(data)
summary = pd.concat(dfs)
fs = summary.groupby([summary.window, summary.posture, summary.cursor_type]).mean()
fs = fs.reset_index()
parameters = list(fs.columns)
remove_columns = ["Unnamed: 0", "subject_num", "repetition", "target_num"]
for removal in remove_columns:
    parameters.remove(removal)
# %%
for p in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
    a = test_score_parameter(param=p, postures=["WALK"])
    # print(a)
# %%
data = []
for p in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
    data.append(pd.read_csv("ParamRawsummary" + str(p) + ".csv"))
data = pd.concat(data)
output = data.groupby([data.cursor_type, data.parameter]).mean()
# %%
data = read_hololens_data(13, "STAND", "HEAD", 5)
splited_data = split_target(data)
t = 6
temp_data = splited_data[t]
temp_data.reset_index(inplace=True)
temp_data.timestamp -= temp_data.timestamp.values[0]
# plt.plot(temp_data.timestamp,temp_data.cursor_angular_distance);plt.show()
plt.plot(temp_data.timestamp, temp_data.head_forward_y)
plt.show()
# grid_size = 4.5
# point = np.array([
#     (-grid_size, grid_size), (0, grid_size), (grid_size, grid_size),
#     (-grid_size, 0), (0, 0), (grid_size, 0),
#     (-grid_size, -grid_size), (0, -grid_size), (grid_size, -grid_size)
# ])
# point_x = np.array([
#     -grid_size, 0, grid_size,
#     -grid_size, 0, grid_size,
#     -grid_size, 0, grid_size
# ])
# point_y = np.array([
#     grid_size, grid_size, grid_size,
#     0, 0, 0,
#     -grid_size, -grid_size, -grid_size
# ])
# max_distance = grid_size * 4
# temp_data['distances'] = temp_data.apply(
#     lambda x: np.sqrt((point_x - x.horizontal_offset) ** 2 + (point_y - x.vertical_offset) ** 2), axis=1)
# temp_data['s_contribute'] = (1 - temp_data.distances / max_distance).apply(np.clip, args=(0, 1))
# # temp_data = temp_data.assign(score=np.array([0,0,0,0,0,0,0,0,0]))
# temp_data['score'] = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])] * len(temp_data)
# param = 0.025
# for index, row in temp_data.iterrows():
#     if index == 0: continue
#     temp_data['score'][index] = temp_data['score'][index - 1] * (1 - param) + temp_data['s_contribute'][index] * (param)
# temp_data['selected_target'] = temp_data.score.apply(np.argmax)
# temp_data['stick_success'] = temp_data['selected_target'] == 4
# for k, g in itertools.groupby(temp_data.iterrows(),
#                               key=lambda row: row[1]['stick_success']):
#     if k == True:
#         df = pd.DataFrame([r[1] for r in g])
#         print(df.timestamp.values[-1] - df.timestamp.values[0])

# temp_data[['score_0', 'score_1', 'score_2', 'score_3'
#     , 'score_4', 'score_5', 'score_6', 'score_7', 'score_8']] = pd.DataFrame(temp_data.score.tolist(),
#                                                                              index=temp_data.index)
# for c in ['score_0', 'score_1', 'score_2', 'score_3'
#     , 'score_4', 'score_5', 'score_6', 'score_7', 'score_8']:
#     if c == 'score_' + str(4):
#         plt.plot(temp_data.timestamp, temp_data[c], 'r-')
#     else:
#         plt.plot(temp_data.timestamp, temp_data[c], 'k-')
#
# plt.title(str(param))
# plt.show()

# %%
# temp_data = get_one_trial(20,'WALK','EYE',5,2)
# summary = pd.read_csv("BasicRawSummary.csv")
# jumps = summary[summary.error == 'jump']
# for trial in jumps.head(3).iterrows():
#     trial = trial[1]
#     temp_data = get_one_trial(trial.subject_num, trial.posture, trial.cursor_type, trial.repetition, trial.target_num)
#     outlier = list(temp_data[(abs(temp_data.target_horizontal_velocity) > 10 * 57.296)].index)
#     outlier = [x for x in outlier if x > 5]
#     outlier_timestamp = temp_data.iloc[outlier].timestamp.values
#     plt.plot(temp_data.timestamp, temp_data.angle, 'r:')
#     plt.plot(temp_data.timestamp, temp_data.max_angle, 'b:')
#     corr_data = check_loss(temp_data, trial.cursor_type)
#     corr_data['target_horizontal_velocity'] = (
#             corr_data['target_horizontal_angle'].diff(1).apply(correct_angle) / corr_data['timestamp'].diff(1))
#     outlier = list(corr_data[(abs(corr_data.target_horizontal_velocity) > 10 * 57.296)].index)
#     outlier = [x for x in outlier if x > 5]
#     corr_outlier_timestamp = corr_data.iloc[outlier].timestamp.values
#     plt.plot(corr_data.timestamp, corr_data.angle, 'r')
#     plt.plot(corr_data.timestamp, corr_data.max_angle, 'b')
#     plt.title(str(corr_outlier_timestamp))
#     # for ol in outlier_timestamp:
#     #     plt.axvline(ol)
#     for ol in corr_outlier_timestamp:
#         plt.axvline(ol)
#     plt.show()
#
#     # plt.plot(corr_data.timestamp,abs(corr_data.target_horizontal_velocity))
#     plt.plot(corr_data.timestamp, corr_data.target_horizontal_velocity)
#     plt.show()
# %% see eye-errors
# repetitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# postures = ['STAND', 'WALK']
#
# cursorTypes = ['EYE']
# targets = range(9)
# for sub_num in range(24):
#     # for sub_num in [6]:
#     subject_total_count = 0
#     subject_error_eye_count = 0
#     subject_error_invalidate = 0
#     for cursor_type, rep, pos in itertools.product(cursorTypes, repetitions, postures):
#         data = read_hololens_data(sub_num, pos, cursor_type, rep)
#         splited_data = split_target(data)
#         # wide = 'SMALL' if rep in rep_small else 'LARGE'
#         for t in targets:
#             try:
#
#                 temp_data = splited_data[t]
#                 temp_data.reset_index(inplace=True)
#                 temp_data.timestamp -= temp_data.timestamp.values[0]
#                 validate, reason = validate_trial_data(temp_data, cursor_type, pos)
#                 temp_data['check_eye'] = temp_data.latestEyeGazeDirection_x.diff(1)
#                 eye_index = temp_data[temp_data.check_eye == 0].index
#                 invalidate_index = temp_data[temp_data.isEyeTrackingEnabledAndValid == False].index
#                 subject_total_count += len(temp_data.index)
#                 subject_error_eye_count += len(eye_index)
#                 subject_error_invalidate += len(invalidate_index)
#                 # plt.plot(temp_data.timestamp,temp_data.horizontal_offset)
#                 # plt.plot(temp_data.timestamp, temp_data.vertical_offset)
#                 # plt.show()
#                 # plt.plot(temp_data.timestamp,temp_data.direction_x)
#                 # plt.plot(temp_data.timestamp, temp_data.head_forward_x)
#                 # plt.show()
#                 # if len(eye_index)>0:
#                 #     print(sub_num, pos, cursor_type, rep,len(eye_index))
#             except Exception as e:
#                 print(sub_num, pos, cursor_type, rep, e.args)
#     print(sub_num, subject_error_eye_count, subject_total_count, subject_error_invalidate,
#           100 * subject_error_eye_count / subject_total_count, '%',
#           100 * subject_error_invalidate / subject_total_count, '%')

# %%
offsets = visualize_offsets(True)
total_error = 0
total_len = 0
for ct, pos in itertools.product(["EYE", "HAND", "HEAD"], ["STAND", "WALK"]):

    h = offsets[(offsets.posture == pos) & (offsets.cursor_type == ct)]["horizontal"]
    hlist = []
    for i in h.values:
        hlist += list(i)
    v = offsets[(offsets.posture == pos) & (offsets.cursor_type == ct)]["vertical"]
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
        if (abs(hlist[i] - hmean) > 3 * sigmas[(ct, pos, "horizontal")]) or (
            abs(vlist[i] - vmean) > 3 * sigmas[(ct, pos, "vertical")]
        ):
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

# %%
summary = visualize_summary(show_plot=True)
summary.to_csv("BasicRawSummary.csv")
# errors = summary[summary.error.isna() == False]
# print('\nError trial counts')
# print(errors.groupby(errors['error']).subject_num.count())
# %% TESTTEST
import seaborn as sns

summary = pd.read_csv("BasicRawSummary.csv")
summary = summary[summary.repetition >= 4]
ee = summary[summary.error.isna() == False]
summary = summary[summary.error.isna()]
print(ee.groupby(ee.error).count().wide)
bySubjects = summary.groupby(
    [summary.subject_num, summary.cursor_type, summary.posture]
).mean()
bySubjects = bySubjects.reset_index()
bySubjects.to_csv("studyOne_BySubject.csv")
ini = bySubjects[["subject_num", "cursor_type", "posture", "initial_contact_time"]]
ini.to_csv("studyOneInitialContactTime.csv")
# summary=summary[summary.longest_dwell_time <=1.1]
# for c in ['mean_offset', 'std_offset', 'initial_contact_time',
#           'target_in_count', 'longest_dwell_time', ]:
#     sns.boxplot(data=summary, x='cursor_type',
#                 hue='posture',
#                 y=c,
#                 showfliers=False,
#                 showmeans=True,
#                 meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
#                            'markersize': '10'},
#                 )
#     plt.show()
# sns.histplot(summary[summary.posture == 'WALK'], x="longest_dwell_time", hue="cursor_type",
#              multiple="dodge", shrink=.8,bins=25,kde=True,alpha=0.05,
#              cumulative=False, stat="density", element="step", fill=True)
# plt.show()
# %% success rate plot?
d = []
summary = summary[summary.error.isna()]
for dt in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    summary[str(int(dt * 1000))] = summary["longest_dwell_time"] > dt - 2 / 60
    dd = summary.groupby([summary.posture, summary.cursor_type]).mean()[
        str(int(dt * 1000))
    ]
    d.append(dd)
    # print(dt, '\n', )
sR = pd.DataFrame(d)
sR.plot()
plt.show()
# %%
# sns.catplot(x='cursor_type', y='mean_offset', hue='posture', data=summary, kind='box', showfliers=True, showmeans=True,
#                 meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '10'})
sns.boxplot(
    data=summary,
    x="cursor_type",
    y="mean_offset",
    hue="posture",
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "x",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "10",
    },
    palette="Set1",
    width=0.8,
)
plt.show()
# %%basic performance results : maybe table?
fs = summary.groupby([summary.posture, summary.cursor_type, summary.wide]).mean()
fs.to_csv("table_candidate_wide.csv")
fs_stds = summary.groupby([summary.posture, summary.cursor_type, summary.wide]).std()
fs_stds.to_csv("table_candidate_wide_stds.csv")
fs_no_wide = summary.groupby([summary.posture, summary.cursor_type]).mean()
fs_no_wide_std = summary.groupby([summary.posture, summary.cursor_type]).std()
# %% box plots?
import seaborn as sns

cs = [
    "mean_offset",
    "std_offset",
    "initial_contact_time",
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
    sns.catplot(
        x="cursor_type",
        y=c,
        hue="posture",
        data=summary,
        kind="box",
        showfliers=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },
    )
    # sns.displot(data=summary,x=c,hue='cursor_type',col='posture',kind='kde')
    # sns.displot(data=summary,y=c,hue='posture',col='cursor_type',kind='kde')
    plt.show()

# %% RM anova
from scipy.stats import shapiro
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# from bioinfokit.analys import stat
# import pyvttbl as pt
from collections import namedtuple
from statannotations.Annotator import Annotator

# pd.set_option('max_colwidth', 400)
# pd.set_option('max_rows', 99999)
summary = pd.read_csv("bS.csv")
# summary['accuracy'] = (summary.mean_offset_horizontal ** 2 + summary.mean_offset_vertical ** 2).apply(math.sqrt)
for column in [
    # 'mean_offset',
    # 'std_offset',
    "initial_contact_time",
    # 'target_in_count',
    # 'total_time',
    # 'success_trial',
    # 'target_in_total_time',
    # 'target_in_mean_time',
    # 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal',
    # 'std_offset_vertical',
    # 'mean_abs_offset_horizontal', 'mean_abs_offset_vertical',
    # 'std_abs_offset_horizontal', 'std_abs_offset_vertical',
    # 'longest_dwell_time', 'movement_length', 'accuracy'
]:

    # s = summary[summary.error.isna()]
    # # ss = s
    #
    # # list_subjects = []
    # # for i, subject_data in s.groupby(s.subject_num):
    # #     d = subject_data[np.abs(stats.zscore(subject_data[column])) < 3]
    # #     list_subjects.append(d.reset_index())
    # # s = pd.concat(list_subjects)
    # # summary=summary[summary.success_trial==1]
    # bySubjects = summary.groupby([summary.subject_num, summary.cursor_type, summary.posture]).mean()
    # bySubjects = bySubjects.reset_index()
    if column == "initial_contact_time":
        aovrm = AnovaRM(
            summary,
            column,
            "subject",
            within=["cursor", "posture"],
            aggregate_func="mean",
        ).fit()
    else:
        aovrm = AnovaRM(
            summary,
            column,
            "subject",
            within=["cursor", "posture"],
            aggregate_func="mean",
        ).fit()

    aov = pg.rm_anova(
        dv=column,
        within=["cursor", "posture"],
        subject="subject",
        data=summary,
        detailed=True,
        effsize="np2",
        correction=True,
    )
    sph = pg.sphericity(
        dv=column,
        within=["cursor", "posture"],
        subject="subject",
        data=summary,
    )
    pg_posthoc = pg.pairwise_ttests(
        dv=column,
        within=["posture", "cursor"],
        subject="subject",
        data=summary,
        padjust="bonf",
    )

    print("METRIC", column)
    # plt.subplot(1, 2, 1)
    # sns.pointplot(data=s[s.posture == 'STAND'], x='cursor_type', y=column, hue='wide', dodge=True,
    #               linestyles=["-", "--"], order=["HAND", 'HEAD', 'EYE'])
    # plt.subplot(1, 2, 2)
    # sns.pointplot(data=s[s.posture == "WALK"], x='cursor_type', y=column, hue='wide', dodge=True,
    #               linestyles=["-", "--"], order=["HAND", 'HEAD', 'EYE'])
    # plt.show()
    order = ["STAND", "WALK"]
    hue_order = ["HAND", "HEAD", "EYE"]
    # pairs = [(('STAND', 'HAND'), ('WALK', 'HAND')),
    #          (('STAND', 'HEAD'), ('WALK', 'HEAD'))]
    pairs = [
        (("STAND", "HAND"), ("STAND", "HEAD")),
        (("STAND", "HAND"), ("STAND", "EYE")),
        (("STAND", "HEAD"), ("STAND", "EYE")),
        (("WALK", "HAND"), ("WALK", "HEAD")),
        (("WALK", "HAND"), ("WALK", "EYE")),
        (("WALK", "HEAD"), ("WALK", "EYE")),
    ]
    ax = sns.boxplot(
        data=bySubjects,
        x="posture",
        y=column,
        hue="cursor_type",
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "x",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },
        palette="Set1",
        width=0.8,
    )
    # annot = Annotator(ax, pairs, data=bySubjects, x='posture', y=column, order=order, hue='cursor_type',
    #                   hue_order=hue_order)
    # annot.new_plot(ax, pairs, data=s, x='posture', y=column, order=order, hue='cursor_type', hue_order=hue_order)
    # annot.configure(test='t-test_ind', comparisons_correction="Bonferroni").apply_test().annotate()
    # sns.boxplot(data=ss, x='posture', y=column, hue='cursor_type', showfliers=False, showmeans=True, )
    plt.legend(loc="upper left", bbox_to_anchor=(1.03, 1), title="Cursor Type")
    plt.title(column)
    plt.tight_layout()
    plt.show()
    from statsmodels.graphics.factorplots import interaction_plot

    # fig = interaction_plot(x=s['posture'], trace=s['cursor_type'], response=s[column],
    #                        colors=['#4c061d', '#d17a22', '#b4c292'])
    # plt.show()
    aov.round(3)
    print("RM-anova(statsmodel)")
    print(aovrm)
    print("RM-anova(pg)")
    # print(aov)
    pg.print_table(aov)
    print("Sphericity(pg)")
    print(sph)
    # plt.figure(figsize=(20, 10))
    # bySubjects.boxplot(column=column,by=['cursor_type','posture'])
    # sns.boxplot(data=bySubjects, x='cursor_type', y=column, hue='posture')
    # plt.show()
    # display(bySubjects)
    bySubjects_wide = s.groupby(
        [s.subject_num, s.cursor_type, s.posture, s.wide]
    ).mean()
    bySubjects_wide = bySubjects_wide.reset_index()
    for pos in ["STAND", "WALK"]:
        print(
            pos,
            bySubjects_wide[bySubjects_wide.posture == pos]
            .pairwise_tukey(dv=column, between=["wide"])
            .round(3),
        )
        print(
            pg.normality(
                data=bySubjects[bySubjects.posture == pos],
                dv=column,
                group="cursor_type",
            )
        )
        # posthoc = pairwise_tukeyhsd(bySubjects[column], bySubjects['cursor_type'], alpha=0.05)
        # print('Posthoc(tukeyhsd)',pos)
        # print(posthoc)
        # fig = posthoc.plot_simultaneous()
        # fig.show()

    print("Posthoc(pairwise ttest,pg)")
    pg.print_table(pg_posthoc)
    # # final summary
    for i in aovrm.anova_table[aovrm.anova_table["Pr > F"] < 0.05].index:
        print("rANOVA significant difference in ", i)
    for i in aov[aov["p-GG-corr"] < 0.05].Source.values:
        print("rANOVA significant difference in (withouf wide)", i)
    print("sphericity:", sph.spher, "p:", sph.pval)
    # print(pg_posthoc)
    print("-" * 100)

# %%dwell-wise anaylsis
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dwell_times = [0.1]
for dt in dwell_times:
    dwell_time_analysis(dt)
# %% visualize dwell-wise analysis
# 527          527         0.1  ...        170.893736   NaN
# 3732        3732         0.1  ...        189.193031   NaN
# 4088        4088         0.1  ...        146.179401   NaN
# 5064        5064         0.1  ...        178.581023   NaN
# 7654
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dwell_time=[0.1]
dfs = []
for dt in dwell_times:
    f = pd.read_csv("dwell_time_Rawsummary" + str(dt) + ".csv")
    f["dwell_time"] = dt
    dfs.append(f)
dwell_summary = pd.concat(dfs)
dwell_summary["success_trial"] = dwell_summary.best_record > dwell_summary.dwell_time
dwell_summary["trial_time"] = dwell_summary.first_dwell_time + dwell_summary.dwell_time
errors = dwell_summary[dwell_summary.error.isna() == False]
# dwell_summary = dwell_summary[dwell_summary.error.isna() == True]
bySubjects = dwell_summary.groupby(
    [
        dwell_summary.dwell_time,
        dwell_summary.subject_num,
        dwell_summary.cursor_type,
        dwell_summary.posture,
    ]
).mean()
bySubjects = bySubjects.reset_index()
plot_df = pd.DataFrame(
    columns=[
        "posture",
        "cursor_type",
        "dwell_time",
        "success_rate",
        "required_target_size",
        "first_dwell_time",
        "final_speed",
        "required_target_size_std",
        "first_dwell_time_std",
        "mean_final_speed_std",
        "best_record",
    ]
)
# for pos, ct in itertools.product(['WALK', 'STAND'], ['HEAD', 'EYE', 'HAND']):
#
#     srs = []
#     for dt in dwell_times:
#         temp = dwell_summary[(dwell_summary.dwell_time == dt) & (dwell_summary.cursor_type == ct)]
#         temp = temp[temp.posture == pos]
#         total_count = len(temp)
#         all_error_count = sum(temp.groupby(temp.error).count().posture)
#         if temp.groupby(temp.error).count().posture.__contains__("no success dwell"):
#             fail_count = temp.groupby(temp.error).count().posture['no success dwell']
#         else:
#             fail_count = 0
#         # print(dt,temp.groupby(temp.error).count().posture)
#         success_rate = 1 - fail_count / (total_count - (all_error_count - fail_count))
#         print(dt, pos, ct, success_rate, total_count, fail_count, all_error_count)
#         print(temp.groupby(temp.error).count().posture)
#         required_target_size = temp.required_target_size.mean()
#         first_dwell_time = temp.first_dwell_time.mean()
#         mean_final_speed = temp.mean_final_speed.mean()
#         mean_best_record = temp.best_record.mean()
#         # errorbar
#         required_target_size_std = temp.required_target_size.std()
#         first_dwell_time_std = temp.first_dwell_time.std()
#         mean_final_speed_std = temp.mean_final_speed.std()
#
#         plot_summary = {'posture': pos,
#                         'cursor_type': ct,
#                         'dwell_time': dt,
#                         'success_rate': success_rate * 100,
#                         'required_target_size': required_target_size,
#                         'first_dwell_time': first_dwell_time,
#                         'mean_final_speed': mean_final_speed,
#                         'required_target_size_std': required_target_size_std,
#                         'first_dwell_time_std': first_dwell_time_std,
#                         'mean_final_speed_std': mean_final_speed_std,
#                         'mean_best_record': mean_best_record
#                         }
#         plot_df.loc[len(plot_df)] = plot_summary
dwell_summary.to_csv("DwellRawSummary.csv")

abc = dwell_summary[
    (dwell_summary.posture == "STAND")
    & (dwell_summary.cursor_type == "HAND")
    & (dwell_summary.dwell_time == 1.0)
]
ddd = dwell_summary[
    (dwell_summary.posture == "STAND")
    & (dwell_summary.cursor_type == "EYE")
    & (dwell_summary.dwell_time == 1.0)
]

bySubjects[
    [
        "dwell_time",
        "subject_num",
        "cursor_type",
        "posture",
        "success_trial",
        "required_target_size",
        "trial_time",
        "final_speed",
        "target_in_count",
    ]
].to_csv("StudyOneResultCollection.csv")
# %% DWELL ANALYSIS
dwell_summary = pd.read_csv("DwellRawSummary.csv")
dwell_summary["success_trial"] = dwell_summary["success_trial"].apply(int)
for column in [
    "success_trial",
    "required_target_size",
    "target_in_count",
    "trial_time",
    "final_speed",
]:
    for pos in [
        "STAND",
        # 'WALK'
    ]:
        aov = pg.rm_anova(
            dv=column,
            within=["cursor_type", "dwell_time"],
            subject="subject_num",
            data=dwell_summary[dwell_summary.posture == pos],
            detailed=True,
            effsize="np2",
            correction=True,
        )
        # for dt in dwell_summary.dwell_time.unique():
        # aov = pg.rm_anova(dv=column, within=['cursor_type'],
        #                   subject='subject_num', data=dwell_summary[(dwell_summary.posture == pos)&(dwell_summary.dwell_time == dt)], detailed=True,
        #                   effsize="np2", correction=True)
        print(column, pos, "RM-anova(pg)")
        pg.print_table(aov)
        # pg_posthoc = pg.pairwise_ttests(dv=column, within=[ 'cursor_type'], subject='subject_num',
        #                                 data=dwell_summary[dwell_summary.posture == pos],
        #                                 padjust='bonf')
        # # print(wt)

        # pg.print_table(pg_posthoc)
# %%
import seaborn as sns

for c in [
    # 'success_trial', 'required_target_size', 'trial_time',
    "final_speed",
    "target_in_count",
]:
    bySubjects[["dwell_time", "subject_num", "cursor_type", "posture", c]].to_csv(
        f"StudyOne_{c}.csv"
    )
    g = sns.catplot(
        kind="bar",
        data=bySubjects,
        col="dwell_time",
        x="cursor_type",
        y=c,
        ci=None,
        # order=['HEAD', 'NEWSPEED', 'NEWSTICKY', 'NEWSPEEDSTICKY'],
        # ci='sd',
        # capsize=.2,
        # palette='husl',
        dodge=True,
        hue="posture",
    )
    plt.show()

# for c in ['success_rate', 'required_target_size', 'first_dwell_time',
#           'mean_final_speed', 'best_record']:
#     fig = go.Figure()
#     for pos, ct in itertools.product(['STAND', 'WALK'], ['HEAD', 'EYE', 'HAND']):
#         plot_data = plot_df[(plot_df.posture == pos) & (plot_df.cursor_type == ct)]
#         # print(plot_data)
#         erry = plot_data[c + '_std'] if c != 'success_rate' else None
#         fig.add_trace(go.Bar(
#             name=pos + '_' + ct, x=dwell_times, y=plot_data[c], error_y=dict(type='data', array=erry)
#         ))
#         fig.update_layout(title=str(c))
#     fig.show()
# %%dwell based plots
# success rate plot
import seaborn as sns

sns.set_style("ticks")
sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(
    x="dwell_time",
    y="success_rate",
    hue="cursor_type",
    data=plot_df,
    palette="Set2",
    marker="o",
    style="posture",
    ci=0,
    style_order=["STAND", "WALK"],
    ax=ax,
)
# sns.lineplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary,
#
#              palette='Set2', marker='o', style='posture', ci=0,
#              # width=0.8,
#              ax=ax)
plt.ylim(0, 110)

plt.xticks(dwell_times)
plt.xlabel("Dwell Threshold (s)")
plt.ylabel("Success Rate (%)")
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles,
    ["Cursor Type", "Head", "Eye", "Hand", "Posture", "STAND", "WALK"],
    loc="center right",
)
plt.tight_layout()
plt.show()
# %% dwell time box plots
dcs = [
    # 'required_target_size',
    "first_dwell_time",
    # 'final_speed',
    # 'min_target_size', 'best_record'
]

sns.set_style("ticks")
sns.set_context("talk")
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
    # sns.lineplot(x='dwell_time', y=c, hue='cursor_type', data=dwell_summary,
    #              # showfliers=False, showmeans=True,
    #              # meanprops={'marker': 'x', 'markerfacecolor': 'white', 'markeredgecolor': 'black',
    #              #            'markersize': '10'},
    #              palette='Set2', marker='o', style='posture', ci=0,
    #              # width=0.8,
    #              ax=ax)
    sns.catplot(
        x="dwell_time",
        y=c,
        hue="cursor_type",
        data=dwell_summary[dwell_summary.posture == "WALK"],
        kind="box",
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "+",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5",
        },
        height=5,
        aspect=2,
        legend=False,
    )
    sns.catplot(
        x="dwell_time",
        y=c,
        hue="cursor_type",
        data=dwell_summary[dwell_summary.posture == "STAND"],
        kind="box",
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "+",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5",
        },
        height=5,
        aspect=2,
        legend=False,
    )
    # sns.displot(data=summary,x=c,hue='cursor_type',col='posture',kind='kde')
    # sns.displot(data=summary,y=c,hue='posture',col='cursor_type',kind='kde')
    if c == "required_target_size":
        plt.ylabel("Required Target Size (°)")
        plt.xticks(dwell_summary.dwell_time.unique())
        plt.ylim(0, 10)
    elif c == "first_dwell_time":
        plt.ylabel("First Dwell Success Time (s)")
        # plt.ylim(0, 5)
    elif c == "final_speed":
        plt.ylabel("Final Cursor Speed (°/s)")
        plt.ylim(0, 20)
    # plt.legend(loc='upper right')
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        ["Cursor Type", "Head", "Eye", "Hand", "Posture", "STAND", "WALK"],
        # loc='center right'
        loc="best",
    )

    # plt.legend(loc='lower left',
    #            bbox_to_anchor=(1, 0),
    # #            fancybox=True, ncol=3,
    #            )
    plt.xlabel("dwell threshold (s)")
    plt.tight_layout()
    plt.show()

# %%
fig = px.bar(
    plot_df,
    x="dwell_time",
    y=["success_rate", "required_target_size", "first_dwell_time", "mean_final_speed"],
    barmode="group",
    facet_col="posture",
    facet_row="cursor_type",
    title="target_size",
)
fig.show()
# %%
fs = summary.groupby([summary.dwell_time, summary.posture, summary.cursor_type]).mean()
fs = fs.reset_index()
parameters = [
    "first_dwell_time",
    "target_in_count",
    "target_in_total_time",
    "target_in_mean_time",
    "required_target_size",
]
fig = px.bar(
    fs,
    x="dwell_time",
    y=parameters,
    barmode="group",
    facet_col="posture",
    facet_row="cursor_type",
    title="target_size",
)
fig.show()
# walks=fs[fs.posture=='WALK']
# %% visualize target-size wise analysis
target_sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

dfs = []
for ts in target_sizes:
    f = pd.read_csv("target_size_summary" + str(ts) + ".csv")
    f["target_size"] = ts
    dfs.append(f)
summary = pd.concat(dfs)
fs = summary.groupby([summary.posture, summary.cursor_type, summary.target_size]).mean()
fs.to_csv("target_size_summary.csv")
fs = pd.read_csv("target_size_summary.csv")
parameters = [
    "initial_contact_time",
    "target_in_count",
    "target_in_total_time",
    "target_in_mean_time",
]
fig = px.bar(
    fs,
    x="target_size",
    y=parameters,
    barmode="group",
    facet_col="posture",
    facet_row="cursor_type",
    title="target_size",
)
fig.show()
# %% fail count
dwell_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
target_sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
dfs = []
for ts in target_sizes:
    f = pd.read_csv("target_size_Rawsummary" + str(ts) + ".csv")
    # f['target_size'] = ts
    # print(f.groupby(f.error).count(),ts)
    fail_count = f.groupby(f.error).count().values[0][0]
    all_count = len(f)
    print(ts, fail_count, "/", all_count, 100 * fail_count / all_count, "%")
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
    c["dwell_time"] = dt
    # df.reset_index(level=0, inplace=True)
    dfs.append(c)

    # print(ss.longest_dwell.count())
# fig = px.bar(fs, x='target_size', y=parameters, barmode='group', facet_col='posture', facet_row='cursor_type',
#              title='target_size')
fs = pd.concat(dfs)
fig = go.Figure()
for cursor_type in ["HEAD", "HAND", "EYE"]:
    for ts in target_sizes:
        for dt in dwell_times:
            fff = fs[
                (fs.cursor_type == cursor_type)
                & (fs.posture == "WALK")
                & (fs.target_size == ts)
                & (fs.dwell_time == dt)
            ]
            fig.add_trace(go.Bar(x=fff.dwell_time, y=fff.longest_dwell))
# fig = px.bar(fs,x='dwell_time',y=['longest_dwell','target_size'],barmode='group', facet_col='posture', facet_row='cursor_type')
# fs = fs.reset_index()
# fig  = px.bar(fs,)
# fig = go.Figure()
fig.show()
# %% pilot study: with/without eye cursor visual

# data = read_hololens_data(0, 'WALK', 'HEAD', t,False)
without_stand = summarize_subject(
    0, ["EYE"], ["STAND"], range(9), [4, 5, 6, 7, 8, 9], True, False
)
with_stand = summarize_subject(
    23, ["EYE"], ["STAND"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
without_walk = summarize_subject(
    1, ["EYE"], ["WALK"], range(9), [4, 5, 6, 7, 8, 9], True, False
)
with_walk = summarize_subject(
    23, ["EYE"], ["WALK"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
print(
    "stand,accuracy",
    with_stand.mean_offset.mean(),
    "->",
    without_stand.mean_offset.mean(),
)
print(
    "walk,accuracy", with_walk.mean_offset.mean(), "->", without_walk.mean_offset.mean()
)
print(
    "stand,precision",
    with_stand.std_offset.mean(),
    "->",
    without_stand.std_offset.mean(),
)
print(
    "walk,precision", with_walk.std_offset.mean(), "->", without_walk.std_offset.mean()
)

i = 2
print(i)
without_stand = summarize_subject(
    i, ["EYE"], ["STAND"], range(9), [4, 5, 6, 7, 8, 9], True, False
)
with_stand = summarize_subject(
    i, ["EYE"], ["STAND"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
without_walk = summarize_subject(
    i, ["EYE"], ["WALK"], range(9), [4, 5, 6, 7, 8, 9], True, False
)
with_walk = summarize_subject(
    i, ["EYE"], ["WALK"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
print(
    "stand,accuracy",
    with_stand.mean_offset.mean(),
    "->",
    without_stand.mean_offset.mean(),
)
print(
    "walk,accuracy", with_walk.mean_offset.mean(), "->", without_walk.mean_offset.mean()
)
print(
    "stand,precision",
    with_stand.std_offset.mean(),
    "->",
    without_stand.std_offset.mean(),
)
print(
    "walk,precision", with_walk.std_offset.mean(), "->", without_walk.std_offset.mean()
)

i = 4
j = 6

print(i)
without_stand = summarize_subject(
    i, ["EYE"], ["STAND"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
with_stand = summarize_subject(
    j, ["EYE"], ["STAND"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
without_walk = summarize_subject(
    i, ["EYE"], ["WALK"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
with_walk = summarize_subject(
    j, ["EYE"], ["WALK"], range(9), [4, 5, 6, 7, 8, 9], False, False
)
print(
    "stand,accuracy",
    with_stand.mean_offset.mean(),
    "->",
    without_stand.mean_offset.mean(),
)
print(
    "walk,accuracy", with_walk.mean_offset.mean(), "->", without_walk.mean_offset.mean()
)
print(
    "stand,precision",
    with_stand.std_offset.mean(),
    "->",
    without_stand.std_offset.mean(),
)
print(
    "walk,precision", with_walk.std_offset.mean(), "->", without_walk.std_offset.mean()
)

# %%
subjective = pd.read_csv("Subjective_study1.csv")
sns.scatterplot(
    data=subjective[subjective.posture == "WALK"],
    x="subject_num",
    y="Temporal",
    hue="cursor_type",
)
plt.show()
sj = subjective.groupby([subjective.posture, subjective.cursor_type]).mean().round(3)
sjt = subjective.groupby([subjective.posture, subjective.cursor_type]).std().round(3)
# r
aov = pg.rm_anova(
    dv="borg",
    within=["cursor_type", "posture"],
    subject="subject_num",
    data=subjective,
    detailed=True,
    effsize="np2",
    correction=True,
)
# aov = pg.anova(dv='borg',between=['posture','cursor_type'],data=subjective,detailed=True)
print(aov.round(3))
ph = pg.pairwise_ttests(data=subjective, dv="borg", between="posture")
print(ph.round(3))
aov = pg.rm_anova(
    dv="nasa",
    within=["cursor_type", "posture"],
    subject="subject_num",
    data=subjective,
    detailed=True,
    effsize="np2",
    correction=True,
)
# aov = pg.anova(dv='borg',between=['posture','cursor_type'],data=subjective,detailed=True)
print(aov.round(3))
ph = pg.pairwise_ttests(data=subjective, dv="nasa", between="posture")
print(ph.round(3))
ph = pg.pairwise_ttests(data=subjective, dv="nasa", between="cursor_type")
print(ph.round(3))
# subjective = subjective.T
# subjective = subjective.reset_index()
# subjective = subjective.reset_index()
# subjective.columns = subjective.columns.droplevel()
# sJ = subjective.unstack()
# sJ= sJ.reset_index(level='subject_num')
# sJ.index=sJ.index.droplevel()
#
# print(sJ)
