import math
import time
from sklearn.decomposition import PCA
import matplotlib.patches
import pandas as pd
import traceback
from FileHandling import *

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from statsmodels.stats.anova import AnovaRM
import pingouin as pg


# matplotlib.use('TkAgg')


def collect_offsets(
    sub_num, cursorTypes=None, postures=None, targets=range(9), repetitions=None
):
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    if postures is None:
        postures = ["STAND", "WALK"]
    if cursorTypes is None:
        cursorTypes = ["HEAD", "EYE", "HAND"]
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]
    summary = pd.DataFrame(
        columns=[
            "subject_num",
            "posture",
            "cursor_type",
            "repetition",
            "target_num",
            "wide",
            "horizontal_offset_list",
            "vertical_offset_list",
            "error",
        ]
    )
    for cursor_type, rep, pos in itertools.product(cursorTypes, repetitions, postures):
        data = read_hololens_data(sub_num, pos, cursor_type, rep)
        splited_data = split_target(data)
        wide = "SMALL" if rep in rep_small else "LARGE"
        for t in targets:
            try:
                trial_summary = {
                    "subject_num": sub_num,
                    "posture": pos,
                    "cursor_type": cursor_type,
                    "repetition": rep,
                    "target_num": t,
                    "wide": wide,
                }
                temp_data = splited_data[t]
                temp_data.reset_index(inplace=True)
                temp_data.timestamp -= temp_data.timestamp.values[0]
                validate, reason = validate_trial_data(temp_data, cursor_type, pos)
                if not validate:  # in case of invalid trial.
                    trial_summary["error"] = reason
                    summary.loc[len(summary)] = trial_summary
                    continue
                drop_index = temp_data[
                    (temp_data["direction_x"] == 0)
                    & (temp_data["direction_y"] == 0)
                    & (temp_data["direction_z"] == 0)
                ].index
                # if len(drop_index) > 0:
                #     print('drop length', len(drop_index), sub_num, pos, cursor_type, rep, t)
                #     # raise ValueError
                temp_data = temp_data.drop(drop_index)
                only_success = temp_data[temp_data.target_name == "Target_" + str(t)]
                if len(only_success) <= 0:
                    raise ValueError("no success frames", len(only_success))
                initial_contact_time = only_success.timestamp.values[0]
                dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                trial_summary["horizontal_offset_list"] = list(
                    dwell_temp.horizontal_offset
                )
                trial_summary["vertical_offset_list"] = list(dwell_temp.vertical_offset)
                summary.loc[len(summary)] = trial_summary
            except Exception as e:
                error_summary = {
                    "subject_num": sub_num,
                    "posture": pos,
                    "cursor_type": cursor_type,
                    "repetition": rep,
                    "target_num": t,
                    "wide": wide,
                    "error": e.args,
                }
                summary.loc[len(summary)] = error_summary
    # final_summary = summary.groupby([summary['posture'], summary['cursor_type'], summary['wide']]).mean()
    # final_summary.to_csv('summary' + str(sub_num) + '.csv')
    summary.to_pickle("offset_lists" + str(sub_num) + ".pkl")
    return summary


def plot_wobble():
    subjects = range(24)
    dfs = []
    for subject in subjects:
        summary_subject = pd.read_pickle("offset_lists" + str(subject) + ".pkl")
        dfs.append(summary_subject)
    summary = pd.concat(dfs)
    summary = summary[summary.error.isna() == True]
    cursorTypes = ["HEAD", "EYE", "HAND"]
    postures = ["STAND", "WALK"]
    summary_dataframe = pd.DataFrame(
        columns=[
            "posture",
            "cursor_type",
            "wide",
            "horizontal",
            "vertical",
            "target_num",
        ]
    )
    for posture, cursor_type in itertools.product(postures, cursorTypes):
        temp_summary = summary[
            (summary.posture == posture) & (summary.cursor_type == cursor_type)
        ]
        hs = []
        vs = []

        for h in temp_summary.horizontal_offset_list.values:
            try:
                hs += h
            except Exception as e:
                pass
                # print(e.args,h)
        for v in temp_summary.vertical_offset_list.values:
            try:
                vs += v
            except Exception as e:
                pass
                # print(e.args,v)
        hs = np.array(hs)
        vs = np.array(vs)
        trial_summary = {
            "posture": posture,
            "cursor_type": cursor_type,
            "horizontal": hs,
            "vertical": vs,
        }
        summary_dataframe.loc[len(summary_dataframe)] = trial_summary
        colorset = [
            "maroon",
            "orangered",
            "darkorange",
            "olive",
            "yellowgreen",
            "darkolivegreen",
            "turquoise",
            "deepskyblue",
            "dodgerblue",
        ]
        postures = ["WALK", "STAND"]
        # postures = ['WALK']
        sigma_multiple = 3
    # for posture, cursor_type in itertools.product(postures, ['EYE', 'HAND', 'HEAD']):
    fig, ax = plt.subplots(
        1, 3, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True
    )
    for idx, cursor_type in enumerate(["HEAD", "EYE", "HAND"]):

        for posture in postures:

            h_raw = summary_dataframe.loc[
                (summary_dataframe["posture"] == posture)
                & (summary_dataframe["cursor_type"] == cursor_type)
            ]["horizontal"].values[0]
            v_raw = summary_dataframe.loc[
                (summary_dataframe["posture"] == posture)
                & (summary_dataframe["cursor_type"] == cursor_type)
            ]["vertical"].values[0]
            h = []
            v = []
            for i in range(len(h_raw)):
                if not (
                    -sigma_multiple * sigmas[(cursor_type, posture, "horizontal")]
                    < h_raw[i]
                    < sigma_multiple * sigmas[(cursor_type, posture, "horizontal")]
                ):
                    continue
                elif not (
                    -sigma_multiple * sigmas[(cursor_type, posture, "vertical")]
                    < v_raw[i]
                    < sigma_multiple * sigmas[(cursor_type, posture, "vertical")]
                ):
                    continue
                h.append(h_raw[i])
                v.append(v_raw[i])
            h = np.array(h)
            v = np.array(v)
            sh = pd.Series(h)
            sh = sh[~((sh - sh.mean()).abs() > 3 * sh.std())]
            sv = pd.Series(v)
            sv = sv[~((sv - sv.mean()).abs() > 3 * sv.std())]
            # sns.kdeplot(x=h + x_offset, y=v + y_offset, fill=True, ax=ax)
            # ax.scatter(x_offset, y_offset, s=100, c=colorset[t], marker='x')

            # confidence_ellipse(x, y, ax_nstd, n_std=1,
            #                    label=r'$1\sigma$', edgecolor='firebrick')
            # confidence_ellipse(x, y, ax_nstd, n_std=2,
            #                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
            # confidence_ellipse(x, y, ax_nstd, n_std=3,
            #                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')
            if posture == "STAND":
                ec = "red"
                ap = 0.03
            else:
                ec = "green"
                ap = 0.05
            ax[idx].scatter(sh, sv, s=0.1, alpha=ap, c=ec)
            plt_confidence_ellipse(
                sh,
                sv,
                ax[idx],
                2,
                linestyle="dotted",
                facecolor="None",
                edgecolor=ec,
                linewidth=3,
            )
            # plt_confidence_ellipse(h, v, ax, 3, linestyle='--', facecolor='None', edgecolor='red',
            #                        linewidth=1)

            # size_ax.scatter(x_offset, y_offset, s=100, c=colorset[t], marker='x')
            # import matplotlib.patches as patches
            # width = h.mean() + 2 * h.std()
            # height = v.mean() + 2 * v.std()
            # size_ax.add_patch(
            #     patches.Rectangle(
            #         ( - width,  - height)
            #         , 2 * width, 2 * height,  fill=False
            #     )
            # )
            # plt.title(str(posture) + "," + str(cursor_type) + '-' + str(sigma_multiple))
        # plt.title(str(posture) + "," + str(cursor_type) )
        circle2 = plt.Circle(
            (0.0, 0.0), 1.5, facecolor="None", edgecolor="blue", linewidth=1
        )
        ax[idx].add_patch(circle2)
        print(ax[idx].patches)
        ax[idx].set_title(str(cursor_type).lower())
        ax[idx].set_xlim(-12, 12)
        ax[idx].set_ylim(-6, 6)
        # plt.ylim(12, 12)
        ax[idx].set_aspect("equal")
        ax[idx].grid()
    plt.show()


def visualize_offsets(show_plot=True):
    subjects = range(24)
    dfs = []
    for subject in subjects:
        summary_subject = pd.read_pickle("offset_lists" + str(subject) + ".pkl")
        dfs.append(summary_subject)
    summary = pd.concat(dfs)
    cursorTypes = ["HEAD", "EYE", "HAND"]
    postures = ["STAND", "WALK"]
    # print('total failure',summary.isnull().mean_offset.sum(),'/',len(summary.index))

    errors = summary[summary.error.isna() == False]
    print(errors.groupby(errors["error"]).subject_num.count())
    # if show_plot:
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, dpi=100)
    # plot over each posture,cursor types
    summary_dict = {}
    summary_dataframe = pd.DataFrame(
        columns=[
            "posture",
            "cursor_type",
            "wide",
            "horizontal",
            "vertical",
            "target_num",
        ]
    )
    for posture, cursor_type, target_num, wide in itertools.product(
        postures, cursorTypes, range(9), ["LARGE", "SMALL"]
    ):
        temp_summary = summary[
            (summary.posture == posture)
            & (summary.cursor_type == cursor_type)
            & (summary.target_num == target_num)
            & (summary.wide == wide)
        ]
        hs = []
        vs = []
        for h in temp_summary.horizontal_offset_list.values:
            try:
                hs += h
            except Exception as e:
                pass
                # print(e.args,h)
        for v in temp_summary.vertical_offset_list.values:
            try:
                vs += v
            except Exception as e:
                pass
                # print(e.args,v)
        hs = np.array(hs)
        vs = np.array(vs)

        trial_summary = {
            "posture": posture,
            "cursor_type": cursor_type,
            "wide": wide,
            "horizontal": hs,
            "vertical": vs,
            "target_num": target_num,
        }
        summary_dataframe.loc[len(summary_dataframe)] = trial_summary

        # summary_dict[(posture, cursor_type, 'horizontal')] = hs
        # summary_dict[(posture, cursor_type, 'vertical')] = vs
    # import seaborn as sns
    # sns.jointplot(x=summary_dict[('STAND','EYE','horizontal')],
    #               y= summary_dict[('STAND','EYE','vertical')],
    #               kind='kde')
    # plt.show()
    if show_plot:

        colorset = [
            "maroon",
            "orangered",
            "darkorange",
            "olive",
            "yellowgreen",
            "darkolivegreen",
            "turquoise",
            "deepskyblue",
            "dodgerblue",
        ]
        postures = ["STAND", "WALK"]
        # postures = ['WALK']
        sigma_multiple = 3
        for posture, cursor_type in itertools.product(
            postures, ["EYE", "HAND", "HEAD"]
        ):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            size_fig, size_ax = plt.subplots(1, 1, figsize=(10, 10))
            for t in range(9):
                for w in ["SMALL", "LARGE"]:
                    if w == "LARGE":
                        wide = 14.04
                    else:
                        wide = 7.125
                    # wide = 10
                    x_offset = wide * math.sin(t * math.pi / 9 * 2)
                    y_offset = wide * math.cos(t * math.pi / 9 * 2)
                    h_raw = summary_dataframe.loc[
                        (summary_dataframe["posture"] == posture)
                        & (summary_dataframe["wide"] == w)
                        & (summary_dataframe["cursor_type"] == cursor_type)
                        & (summary_dataframe["target_num"] == t)
                    ]["horizontal"].values[0]
                    v_raw = summary_dataframe.loc[
                        (summary_dataframe["posture"] == posture)
                        & (summary_dataframe["wide"] == w)
                        & (summary_dataframe["cursor_type"] == cursor_type)
                        & (summary_dataframe["target_num"] == t)
                    ]["vertical"].values[0]
                    h = []
                    v = []
                    for i in range(len(h_raw)):
                        if not (
                            -sigma_multiple
                            * sigmas[(cursor_type, posture, "horizontal")]
                            < h_raw[i]
                            < sigma_multiple
                            * sigmas[(cursor_type, posture, "horizontal")]
                        ):
                            continue
                        elif not (
                            -sigma_multiple * sigmas[(cursor_type, posture, "vertical")]
                            < v_raw[i]
                            < sigma_multiple
                            * sigmas[(cursor_type, posture, "vertical")]
                        ):
                            continue
                        h.append(h_raw[i])
                        v.append(v_raw[i])
                    h = np.array(h)
                    v = np.array(v)
                    # sns.kdeplot(x=h + x_offset, y=v + y_offset, fill=True, ax=ax)
                    ax.scatter(x_offset, y_offset, s=100, c=colorset[t], marker="x")
                    # ax.scatter(h + x_offset, v + y_offset, s=0.5, alpha=0.05, c=colorset[t])
                    plt_confidence_ellipse(
                        h + x_offset,
                        v + y_offset,
                        ax,
                        2,
                        edgecolor=colorset[t],
                        linestyle="--",
                        linewidth=3,
                    )
                    size_ax.scatter(
                        x_offset, y_offset, s=100, c=colorset[t], marker="x"
                    )
                    import matplotlib.patches as patches

                    width = h.mean() + 2 * h.std()
                    height = v.mean() + 2 * v.std()
                    size_ax.add_patch(
                        patches.Rectangle(
                            (x_offset - width, y_offset - height),
                            2 * width,
                            2 * height,
                            edgecolor=colorset[t],
                            fill=False,
                        )
                    )
            plt.title(str(posture) + "," + str(cursor_type) + "-" + str(sigma_multiple))
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.show()
    return summary_dataframe


@timeit
def draw_ellipse_second_study(
    cursorTypes=None,
    targetTypes=None,
    targets=range(8),
    repetitions=None,
    saveFile=True,
):
    if repetitions is None:
        repetitions = [2, 3, 4]

    if cursorTypes is None:
        cursorTypes = ["HEAD", "NEWSPEED", "NEWSTICKY", "NEWSPEEDSTICKY"]
    if targetTypes is None:
        targetTypes = ["GRID", "MENU", "PIE"]
    subjects = range(16)
    summary = pd.DataFrame(
        columns=[
            "cursor_type",
            "target_type",
            "repetition",
            "target_num",
            "coordinates_x",
            "coordinates_y",
            "error",
        ]
    )
    for sub_num, ct, tt, rep, t in itertools.product(
        subjects, cursorTypes, targetTypes, repetitions, targets
    ):
        try:
            print(sub_num, ct, tt, rep, t)
            temp_data = read_second_data(
                subject=sub_num,
                cursor_type=ct,
                repetition=rep,
                target_type=tt,
                target_num=t,
            )
            temp_data.reset_index(inplace=True)
            temp_data.timestamp -= temp_data.timestamp.values[0]
            only_success = temp_data[temp_data.success == True]
            if len(only_success) <= 0:
                raise ValueError("no success frames", len(only_success))
            initial_contact_time = only_success.timestamp.values[0]
            after = temp_data[temp_data.timestamp >= initial_contact_time]
            coordinate_x = after.horizontal_offset.to_list()
            coordinate_y = after.vertical_offset.to_list()
            trial_summary = {
                "cursor_type": ct,
                "target_type": tt,
                "repetition": rep,
                "target_num": t,
                "coordinates_x": coordinate_x,
                "coordinates_y": coordinate_y,
                "error": None,
            }
            summary.loc[len(summary)] = trial_summary
        except Exception as e:
            error_trial_summary = {
                "subject_num": sub_num,
                "cursor_type": ct,
                "target_type": tt,
                "repetition": rep,
                "target_num": t,
                "error": e.args,
            }
            summary.loc[len(summary)] = error_trial_summary
    if saveFile:
        summary.to_pickle("secondOffsets.pkl")
    return summary


@timeit
def summarize_second_study(
    sub_num,
    cursorTypes=None,
    targetTypes=None,
    targets=range(8),
    repetitions=None,
    saveFile=True,
    pickle=False,
):
    if repetitions is None:
        repetitions = [2, 3, 4]

    if cursorTypes is None:
        cursorTypes = ["HEAD", "NEWSPEED", "NEWSTICKY", "NEWSPEEDSTICKY"]
    if targetTypes is None:
        targetTypes = ["GRID", "MENU", "PIE"]
    # if target_nums is None:
    #     target_nums = [2,3,4]
    summary = pd.DataFrame(
        columns=[
            "subject_num",
            "posture",
            "cursor_type",
            "target_type",
            "repetition",
            "target_num",
            "longest_dwell_time",
            "mean_dwell_time",
            "initial_contact_time",
            "mean_offset",
            "total_dwell_time",
            "success_time",
            "std_offset",
            "mean_offset_horizontal",
            "mean_offset_vertical",
            "std_offset_horizontal",
            "std_offset_vertical",
            "success_trial",
            "trial_time",
            "drop_count",
            "abs_mean_offset_horizontal",
            "abs_mean_offset_vertical",
            "drop_out_count",
            "mean_out_time",
            "drop_positions",
            "drop_vectors",
            "walking_speed",
            "straight_length",
            "walk_length",
            "straightness_simple",
            "out_mean_distance",
            "straightness",
            "curve",
            "error",
        ]
    )
    print("summarizing second study : ", sub_num)
    for ct, tt, rep in itertools.product(cursorTypes, targetTypes, repetitions):
        dfs = []
        for t in targets:
            temp_data = read_second_data(
                subject=sub_num,
                cursor_type=ct,
                repetition=rep,
                target_type=tt,
                target_num=t,
            )
            dfs.append(temp_data)
        data = pd.concat(dfs)
        pca = PCA(n_components=2)
        X = data[["head_position_x", "head_position_z"]]
        X = np.array(X)
        pca.fit(X)
        Z = pca.transform(X)
        ZZ = pd.DataFrame(Z)
        ZM = ZZ[ZZ[0] < 0]
        ZP = ZZ[ZZ[0] > 0]
        leftMax = ZM[0].values[ZM[1].argmax()]
        leftMin = ZM[0].values[ZM[1].argmin()]
        lefty = max(leftMax, leftMin)
        rightMax = ZP[0].values[ZP[1].argmax()]
        rightMin = ZP[0].values[ZP[1].argmin()]
        righty = min(rightMin, rightMax)
        for t in targets:
            # temp_data = read_second_data(subject=sub_num, cursor_type=ct, repetition=rep, target_type=tt, target_num=t)
            try:
                trial_summary = {
                    "subject_num": sub_num,
                    "cursor_type": ct,
                    "target_type": tt,
                    "repetition": rep,
                    "target_num": t,
                }
                temp_data = data[data.end_num == t]
                temp_data.reset_index(inplace=True)
                temp_data.timestamp -= temp_data.timestamp.values[0]
                only_success = temp_data[temp_data.success == True]
                if len(only_success) <= 0:
                    raise ValueError("no success frames", len(only_success))
                initial_contact_time = only_success.timestamp.values[0]
                if "STICKY" in ct:
                    temp_data[
                        [
                            "score_0",
                            "score_1",
                            "score_2",
                            "score_3",
                            "score_4",
                            "score_5",
                            "score_6",
                            "score_7",
                        ]
                    ] = pd.DataFrame(temp_data.scores.tolist(), index=temp_data.index)

                success_dwells = []
                fail_dwells = []

                for k, g in itertools.groupby(
                    temp_data.iterrows(), key=lambda row: row[1]["success"]
                ):
                    # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_name']):
                    # print(k, [t[0] for t in g])
                    if k == True:
                        # if k == 'Target_' + str(t):
                        df = pd.DataFrame([r[1] for r in g])
                        success_dwells.append(df)
                    # if k == False:
                    #     df = pd.DataFrame([r[1] for r in g])
                    #     fail_dwells.append(df)
                times = []
                for dw in success_dwells:
                    current_dwell_time = (
                        dw.timestamp.values[-1] - dw.timestamp.values[0]
                    )
                    times.append(current_dwell_time)

                longest_dwell_time = max(times)
                total_dwell_time = sum(times)
                mean_dwell_time = sum(times) / len(times)
                success_time = success_dwells[-1].timestamp.values[0]
                dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                drop_positions = []
                drop_vectors = []

                temp_X = temp_data[["head_position_x", "head_position_z"]]
                temp_X = np.array(temp_X)
                temp_transform = pca.transform(temp_X)
                straightness = np.sum(
                    (lefty < temp_transform[:, 0]) & (temp_transform[:, 0] < righty)
                ) / len(temp_transform)
                if straightness >= 1.0:
                    curve = "straight"
                elif 1 > straightness > 0:
                    curve = "between"
                else:
                    curve = "curve"

                for k, g in itertools.groupby(
                    dwell_temp.iterrows(), key=lambda row: row[1]["success"]
                ):
                    if k == False:
                        df = pd.DataFrame([r[1] for r in g])
                        fail_dwells.append(df)
                        pos = (
                            df.horizontal_offset.values[-1],
                            df.vertical_offset.values[-1],
                        )
                        drop_positions.append(pos)
                        pos = np.array(pos)
                        drop_vectors.append(pos / np.linalg.norm(pos))
                fail_times = []
                out_distances = []
                for fw in fail_dwells:
                    fail_time = fw.timestamp.values[-1] - fw.timestamp.values[0]
                    fail_times.append(fail_time)
                    out_distances.append(fw.cursor_angular_distance.mean())
                    # drop_positions.append((fw.horizontal_offset.values[0], fw.vertical_offset.values[0]))

                mean_offset_horizontal = dwell_temp.horizontal_offset.mean()
                abs_mean_offset_horizontal = dwell_temp.horizontal_offset.apply(
                    abs
                ).mean()
                std_offset_horizontal = dwell_temp.horizontal_offset.std()
                mean_offset_vertical = dwell_temp.vertical_offset.mean()
                abs_mean_offset_vertical = dwell_temp.vertical_offset.apply(abs).mean()
                std_offset_vertical = dwell_temp.vertical_offset.std()
                walklength = (
                    (
                        temp_data.head_position_x.diff(1) ** 2
                        + temp_data.head_position_y.diff(1) ** 2
                        + temp_data.head_position_z.diff(1) ** 2
                    )
                    .apply(math.sqrt)
                    .sum()
                )
                straight_length = math.sqrt(
                    (
                        temp_data.head_position_x.values[0]
                        - temp_data.head_position_x.values[-1]
                    )
                    ** 2
                    + (
                        temp_data.head_position_z.values[0]
                        - temp_data.head_position_z.values[-1]
                    )
                    ** 2
                )
                walking_speed = walklength / (
                    temp_data.timestamp.values[-1] - temp_data.timestamp.values[0]
                )
                # if straight_length ==0: print('straight zero')
                # if len(fail_times)==0 : print('fail zero')
                # if len(times) == 0: print('times zero')

                if len(fail_times) == 0:
                    mean_out_time = None
                else:
                    mean_out_time = sum(fail_times) / len(fail_times)
                trial_summary = {
                    "subject_num": sub_num,
                    "cursor_type": ct,
                    "target_type": tt,
                    "repetition": rep,
                    "target_num": t,
                    "longest_dwell_time": longest_dwell_time,
                    "total_dwell_time": total_dwell_time,
                    "mean_dwell_time": mean_dwell_time,
                    "success_time": success_time,
                    "initial_contact_time": initial_contact_time,
                    "mean_offset": dwell_temp.angle.mean(),
                    "std_offset": dwell_temp.angle.std(),
                    "mean_offset_horizontal": mean_offset_horizontal,
                    "mean_offset_vertical": mean_offset_vertical,
                    "std_offset_horizontal": std_offset_horizontal,
                    "std_offset_vertical": std_offset_vertical,
                    "success_trial": longest_dwell_time >= 1.0 - 2.5 / 60.0,
                    "trial_time": temp_data.timestamp.values[-1]
                    - temp_data.timestamp.values[0],
                    "drop_count": len(success_dwells),
                    "abs_mean_offset_horizontal": abs_mean_offset_horizontal,
                    "abs_mean_offset_vertical": abs_mean_offset_vertical,
                    "drop_out_count": len(fail_times),
                    "mean_out_time": mean_out_time,
                    "drop_positions": drop_positions,
                    "drop_vectors": drop_vectors,
                    "walking_speed": walking_speed,
                    "straight_length": straight_length,
                    "walk_length": walklength,
                    "straightness_simple": walklength / straight_length,
                    "out_mean_distance": out_distances,
                    "straightness": straightness,
                    "curve": curve,
                    "error": None,
                }
                summary.loc[len(summary)] = trial_summary

                # print('best dwell time:', round(longest_dwell_time, 2), sub_num, ct, rep, tt, t)
            except Exception as e:
                error_trial_summary = {
                    "subject_num": sub_num,
                    "cursor_type": ct,
                    "target_type": tt,
                    "repetition": rep,
                    "target_num": t,
                    "error": e.args,
                }
                summary.loc[len(summary)] = error_trial_summary
                print(sub_num, ct, rep, tt, t, e.args)
    final_summary = summary.groupby(
        [summary["cursor_type"], summary["target_type"]]
    ).mean()
    if saveFile:
        if pickle:
            final_summary.to_pickle("second_summary" + str(sub_num) + ".pkl")
            summary.to_pickle("second_Rawsummary" + str(sub_num) + ".pkl")
        else:
            final_summary.to_csv("second_summary" + str(sub_num) + ".csv")
            summary.to_csv("second_Rawsummary" + str(sub_num) + ".csv")

    return summary, final_summary


# def path_finding(coordinates):


@timeit
def summarize_subject(
    sub_num,
    cursorTypes=None,
    postures=None,
    targets=range(9),
    repetitions=None,
    pilot=False,
    savefile=True,
    resetFile=False,
    secondstudy=False,
    fnc=None,
    suffix="",
    arg=None,
):
    print("summarizing subject", sub_num)
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    if postures is None:
        postures = ["STAND", "WALK"]
    if cursorTypes is None:
        cursorTypes = ["HEAD", "EYE", "HAND"]

    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]
    draw_plot = False

    # repetitions = range(10)
    fail_count = 0
    summary = pd.DataFrame(
        columns=[
            "subject_num",
            "posture",
            "cursor_type",
            "repetition",
            "target_num",
            "wide",
            "mean_offset",
            "std_offset",
            "overall_mean_offset",
            "overall_std_offset",
            "initial_contact_time",
            "target_in_count",
            "target_in_total_time",
            "target_in_mean_time",
            "mean_offset_horizontal",
            "mean_offset_vertical",
            "std_offset_horizontal",
            "std_offset_vertical",
            "mean_abs_offset_horizontal",
            "mean_abs_offset_vertical",
            "std_abs_offset_horizontal",
            "std_abs_offset_vertical",
            "longest_dwell_time",
            "movement_length",
            "entering_position",
            "walking_speed",
            "total_time",
            "success_trial",
            "error",
            "error_frame_count",
        ]
    )
    for cursor_type in cursorTypes:
        for rep in repetitions:
            for pos in postures:
                data = read_hololens_data(
                    sub_num, pos, cursor_type, rep, resetFile, pilot, secondstudy
                )

                # print(data)
                # data=data.apply(fnc,arg)
                # pca = PCA(n_components=2)
                # data['head_position_x'] = data['head_position_x'].interpolate()
                # data['head_position_z'] = data['head_position_z'].interpolate()
                # X = data[['head_position_x', 'head_position_z']]
                # X = np.array(X)
                # pca.fit(X)
                # Z = pca.transform(X)
                # ZZ = pd.DataFrame(Z)
                # ZM = ZZ[ZZ[0] < 0]
                # ZP = ZZ[ZZ[0] > 0]
                # leftMax = ZM[0].values[ZM[1].argmax()]
                # leftMin = ZM[0].values[ZM[1].argmin()]
                # lefty = max(leftMax, leftMin)
                # rightMax = ZP[0].values[ZP[1].argmax()]
                # rightMin = ZP[0].values[ZP[1].argmin()]
                # righty = min(rightMin, rightMax)

                splited_data = split_target(data)
                wide = "SMALL" if rep in rep_small else "LARGE"

                for t in targets:
                    try:
                        trial_summary = {
                            "subject_num": sub_num,
                            "posture": pos,
                            "cursor_type": cursor_type,
                            "repetition": rep,
                            "target_num": t,
                            "wide": wide,
                        }
                        temp_data = splited_data[t]

                        temp_data.reset_index(inplace=True)
                        temp_data.timestamp -= temp_data.timestamp.values[0]
                        if fnc != None:
                            temp_data = fnc(temp_data, arg)
                        # drop_index = output[
                        #     (output['abs_horizontal_offset'] > 3 * sigmas[(cursor_type, posture, 'horizontal')]) | (
                        #             output['abs_vertical_offset'] > 3 * sigmas[(cursor_type, posture, 'vertical')])]
                        temp_data = check_loss(temp_data, cursor_type)
                        trial_summary["error_frame_count"] = len(
                            temp_data[temp_data["error_frame"] == True]
                        )
                        # # temp_data = temp_data.drop(drop_index)
                        validate, reason = validate_trial_data(
                            temp_data, cursor_type, pos
                        )
                        if not validate:  # in case of invalid trial.
                            trial_summary["error"] = reason
                            print(sub_num, pos, cursor_type, rep, t, reason)
                            summary.loc[len(summary)] = trial_summary
                            continue

                        temp_data["cursor_speed"] = temp_data.angle.diff(
                            1
                        ) / temp_data.timestamp.diff(1)
                        temp_data["cursor_speed"] = abs(
                            temp_data.cursor_speed.rolling(
                                10, min_periods=1, center=True
                            ).mean()
                        )

                        # only_success = temp_data[temp_data.cursor_angular_distance < default_target_size]
                        only_success = temp_data[temp_data.success == True]
                        if len(only_success) <= 0:
                            raise ValueError("no success frames", len(only_success))
                        initial_contact_time = only_success.timestamp.values[0]

                        success_dwells = []

                        # if "STICKY" in cursor_type:  # try another approach
                        #     score_columns = ['score' + str(tn) for tn in range(8)]
                        #     scores = pd.DataFrame(temp_data.scores.to_list(), columns=score_columns,
                        #                           index=temp_data.index)
                        #     temp_data = pd.concat([temp_data, scores], axis=1)
                        #     temp_data['selected_target'] = temp_data.scores.apply(np.argmax)
                        #     temp_data['stick_success'] = temp_data['selected_target'] == t
                        #     for k, g in itertools.groupby(temp_data.iterrows(),
                        #                                   key=lambda row: row[1]['stick_success']):
                        #         if k == True:
                        #             df = pd.DataFrame([r[1] for r in g])
                        #             success_dwells.append(df)
                        # else:
                        # temp_data['target_in'] = temp_data['cursor_angular_distance'] < default_target_size
                        for k, g in itertools.groupby(
                            temp_data.iterrows(), key=lambda row: row[1]["success"]
                        ):
                            # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_name']):
                            # print(k, [t[0] for t in g])
                            if k == True:
                                # if k == 'Target_' + str(t):
                                df = pd.DataFrame([r[1] for r in g])
                                success_dwells.append(df)
                        time_sum = 0
                        times = []
                        for dw in success_dwells:
                            current_dwell_time = (
                                dw.timestamp.values[-1] - dw.timestamp.values[0]
                            )
                            time_sum += current_dwell_time
                            times.append(current_dwell_time)

                        target_in_count = len(success_dwells)
                        target_in_total_time = time_sum
                        target_in_mean_time = time_sum / target_in_count
                        # TODO
                        longest_dwell_time = max(times)
                        # if longest_dwell_time >= 1:
                        #     longest_dwell_time = 1.0
                        targeting = temp_data[
                            temp_data.timestamp <= initial_contact_time
                        ]
                        movement = (
                            targeting.horizontal_offset.diff(1) ** 2
                            + targeting.vertical_offset.diff(1) ** 2
                        ).apply(math.sqrt)
                        movement_length = movement.sum()
                        contact_frame = temp_data[
                            temp_data.timestamp == initial_contact_time
                        ]
                        x = contact_frame.horizontal_offset.values[0]
                        y = contact_frame.vertical_offset.values[0]
                        entering_position = (-x, y)

                        dwell_temp = temp_data[
                            temp_data.timestamp >= initial_contact_time
                        ]
                        #
                        mean_offset_horizontal = dwell_temp.horizontal_offset.mean()
                        std_offset_horizontal = dwell_temp.horizontal_offset.std()
                        mean_offset_vertical = dwell_temp.vertical_offset.mean()
                        std_offset_vertical = dwell_temp.vertical_offset.std()
                        walklength = (
                            (
                                temp_data.head_position_x.diff(1) ** 2
                                + temp_data.head_position_z.diff(1) ** 2
                            )
                            .apply(math.sqrt)
                            .sum()
                        )
                        walking_speed = walklength / (
                            temp_data.timestamp.values[-1]
                            - temp_data.timestamp.values[0]
                        )
                        temp_data["head_position_x"] = temp_data[
                            "head_position_x"
                        ].interpolate()
                        temp_data["head_position_z"] = temp_data[
                            "head_position_z"
                        ].interpolate()
                        # temp_X = temp_data[['head_position_x', 'head_position_z']]
                        # temp_X = np.array(temp_X)
                        # temp_transform = pca.transform(temp_X)
                        # straightness = np.sum((lefty < temp_transform[:, 0]) & (temp_transform[:, 0] < righty)) / len(
                        #     temp_transform)
                        # if straightness >= 1.0:
                        #     curve = 'straight'
                        # elif 1 > straightness > 0:
                        #     curve = 'between'
                        # else:
                        #     curve = 'curve'
                        trial_success = longest_dwell_time >= 1.0 - 2.5 / 60
                        trial_summary = {
                            "subject_num": sub_num,
                            "posture": pos,
                            "cursor_type": cursor_type,
                            "repetition": rep,
                            "target_num": t,
                            "wide": wide,
                            "mean_offset": dwell_temp.angle.mean(),
                            "std_offset": dwell_temp.angle.std(),
                            "overall_mean_offset": temp_data.angle.mean(),
                            "overall_std_offset": temp_data.angle.std(),
                            "initial_contact_time": initial_contact_time,
                            "target_in_count": float(target_in_count),
                            "target_in_total_time": target_in_total_time,
                            "target_in_mean_time": target_in_mean_time,
                            "mean_offset_horizontal": mean_offset_horizontal,
                            "mean_offset_vertical": mean_offset_vertical,
                            "std_offset_horizontal": std_offset_horizontal,
                            "std_offset_vertical": std_offset_vertical,
                            "mean_abs_offset_horizontal": dwell_temp.abs_horizontal_offset.mean(),
                            "mean_abs_offset_vertical": dwell_temp.abs_vertical_offset.mean(),
                            "std_abs_offset_horizontal": dwell_temp.abs_horizontal_offset.std(),
                            "std_abs_offset_vertical": dwell_temp.abs_vertical_offset.std(),
                            "longest_dwell_time": longest_dwell_time,
                            "movement_length": movement_length,
                            "entering_position": entering_position,
                            "walking_speed": walking_speed,
                            "total_time": temp_data.timestamp.values[-1],
                            "success_trial": trial_success,
                            # 'curve': curve,
                            "error": None,
                        }
                        summary.loc[len(summary)] = trial_summary
                        # smalls.loc[len(smalls)] = [sub_num, 'STAND', cursor_type, rep, t, temp_data.cursor_angular_distance.mean(),
                        #                            temp_data.cursor_angular_distance.std()]
                    except Exception as e:
                        fail_count += 1
                        error_summary = {
                            "subject_num": sub_num,
                            "posture": pos,
                            "cursor_type": cursor_type,
                            "repetition": rep,
                            "target_num": t,
                            "wide": wide,
                            "error": e.args,
                        }
                        summary.loc[len(summary)] = error_summary
                        print(
                            sub_num,
                            pos,
                            cursor_type,
                            rep,
                            t,
                            e.args,
                            "fail count",
                            fail_count,
                        )
    # print(summary)
    final_summary = summary.groupby(
        [summary["posture"], summary["cursor_type"], summary["wide"]]
    ).mean()
    if savefile:
        if pilot:
            final_summary.to_csv(suffix + "nocursor_summary" + str(sub_num) + ".csv")
            summary.to_csv(suffix + "nocursor_Rawsummary" + str(sub_num) + ".csv")
        elif secondstudy:
            final_summary.to_csv(suffix + "second_summary" + str(sub_num) + ".csv")
            summary.to_csv(suffix + "second_Rawsummary" + str(sub_num) + ".csv")
        else:
            final_summary.to_csv(suffix + "summary" + str(sub_num) + ".csv")
            summary.to_csv(suffix + "Rawsummary" + str(sub_num) + ".csv")
    return summary


@timeit
def test_score_parameter(
    param=0.01,
    subjects=range(24),
    cursorTypes=None,
    postures=None,
    targets=range(9),
    repetitions=None,
    pilot=False,
    savefile=True,
    resetFile=False,
    secondstudy=False,
):
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    if postures is None:
        postures = ["STAND", "WALK"]
    if cursorTypes is None:
        cursorTypes = ["HEAD", "EYE", "HAND"]
    grid_size = 4.5
    point = np.array(
        [
            (-grid_size, grid_size),
            (0, grid_size),
            (grid_size, grid_size),
            (-grid_size, 0),
            (0, 0),
            (grid_size, 0),
            (-grid_size, -grid_size),
            (0, -grid_size),
            (grid_size, -grid_size),
        ]
    )
    point_x = np.array(
        [-grid_size, 0, grid_size, -grid_size, 0, grid_size, -grid_size, 0, grid_size]
    )
    point_y = np.array(
        [grid_size, grid_size, grid_size, 0, 0, 0, -grid_size, -grid_size, -grid_size]
    )
    max_distance = grid_size * 4
    summary = pd.DataFrame(
        columns=[
            "subject_num",
            "posture",
            "cursor_type",
            "repetition",
            "target_num",
            "parameter",
            "target_in_count",
            "time_sum",
            "target_in_total_time",
            "target_in_mean_time",
            "longest_dwell_time",
            "initial_contact_time",
            "error",
            "error_frame_count",
        ]
    )
    for sub_num, cursor_type, rep, pos in itertools.product(
        subjects, cursorTypes, repetitions, postures
    ):
        # for sub_num in subjects:
        #     for cursor_type in cursorTypes:
        #         for rep in repetitions:
        #             for pos in postures:
        data = read_hololens_data(
            sub_num, pos, cursor_type, rep, resetFile, pilot, secondstudy
        )
        splited_data = split_target(data)

        for t in targets:
            try:
                trial_summary = {
                    "subject_num": sub_num,
                    "posture": pos,
                    "cursor_type": cursor_type,
                    "repetition": rep,
                    "target_num": t,
                    "parameter": param,
                    # 'wide': wide,
                }
                temp_data = splited_data[t]

                temp_data.reset_index(inplace=True)
                temp_data.timestamp -= temp_data.timestamp.values[0]
                temp_data = check_loss(temp_data, cursor_type)
                trial_summary["error_frame_count"] = len(
                    temp_data[temp_data["error_frame"] == True]
                )
                # # temp_data = temp_data.drop(drop_index)
                validate, reason = validate_trial_data(temp_data, cursor_type, pos)

                if not validate:  # in case of invalid trial.
                    trial_summary["error"] = reason
                    print(sub_num, pos, cursor_type, rep, t, reason)
                    summary.loc[len(summary)] = trial_summary
                    continue
                temp_data["distances"] = temp_data.apply(
                    lambda x: np.sqrt(
                        (point_x - x.horizontal_offset) ** 2
                        + (point_y - x.vertical_offset) ** 2
                    ),
                    axis=1,
                )
                temp_data["s_contribute"] = (
                    1 - temp_data.distances / max_distance
                ).apply(np.clip, args=(0, 1))
                # temp_data = temp_data.assign(score=np.array([0,0,0,0,0,0,0,0,0]))
                temp_data["score"] = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])] * len(
                    temp_data
                )
                for index, row in temp_data.iterrows():
                    if index == 0:
                        continue
                    temp_data["score"][index] = temp_data["score"][index - 1] * (
                        1 - param
                    ) + temp_data["s_contribute"][index] * (param)
                temp_data["selected_target"] = temp_data.score.apply(np.argmax)
                temp_data["stick_success"] = temp_data["selected_target"] == 4
                dwell_times = []
                for k, g in itertools.groupby(
                    temp_data.iterrows(), key=lambda row: row[1]["stick_success"]
                ):
                    if k == True:
                        df = pd.DataFrame([r[1] for r in g])
                        dwell_times.append(
                            df.timestamp.values[-1] - df.timestamp.values[0]
                        )
                        # print(df.timestamp.values[-1] - df.timestamp.values[0])
                target_in_count = len(dwell_times)
                time_sum = sum(dwell_times)
                target_in_total_time = time_sum
                target_in_mean_time = time_sum / target_in_count
                longest_dwell_time = max(dwell_times)
                initial_contact_time = temp_data[
                    temp_data.stick_success == True
                ].timestamp.values[0]
                trial_summary = {
                    "subject_num": sub_num,
                    "posture": pos,
                    "cursor_type": cursor_type,
                    "repetition": rep,
                    "target_num": t,
                    "parameter": param,
                    "target_in_count": target_in_count,
                    "time_sum": time_sum,
                    "target_in_total_time": target_in_total_time,
                    "target_in_mean_time": target_in_mean_time,
                    "longest_dwell_time": longest_dwell_time,
                    "initial_contact_time": initial_contact_time,
                    # 'wide': wide,
                }
                summary.loc[len(summary)] = trial_summary
            except Exception as e:
                error_summary = {
                    "subject_num": sub_num,
                    "posture": pos,
                    "cursor_type": cursor_type,
                    "repetition": rep,
                    "target_num": t,
                    "parameter": param,
                    # 'wide': wide,
                    "error": e.args,
                }
                summary.loc[len(summary)] = error_summary
                # print(sub_num, pos, cursor_type, rep, t, e.args)
    final_summary = summary.groupby([summary["posture"], summary["cursor_type"]]).mean()
    if savefile:
        final_summary.to_csv("Paramsummary" + str(param) + ".csv")
        summary.to_csv("ParamRawsummary" + str(param) + ".csv")
        print("saving file...", "Paramsummary" + str(param) + ".csv")

    return summary


def visualize_summary(
    show_plot=True, show_distribution=False, subjects=range(24), suffix=""
):
    # subjects = range(24)
    dfs = []
    for subject in subjects:
        summary_subject = pd.read_csv(suffix + "Rawsummary" + str(subject) + ".csv")
        dfs.append(summary_subject)
    summary = pd.concat(dfs)
    errors = summary[summary.error.isna() == False]
    print(errors.groupby(errors["error"]).subject_num.count())

    def cart2pol(x, y):
        z = x + y * 1j
        r, theta = np.abs(z), np.angle(z)
        return (r, theta)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    # from ast import literal_eval
    # for pos in ['STAND']:
    #     for t in range(9):
    #         entering = summary[(summary.target_num == t) & (summary.posture == pos)].entering_position
    #         a = entering[entering.notna()]
    #         entering_positions = a.apply(literal_eval).values
    #         x, y = zip(*entering_positions)
    #         spherical = list(map(cart2pol, x, y))
    #         sx, sy = zip(*spherical)
    #         fig = plt.figure()
    #         ax = fig.add_subplot(projection='polar')
    #
    #         colors = sy
    #         c = ax.scatter(sy, sx, marker='.', alpha=0.75)
    #
    #         default_r, default_theta = cart2pol(directions[t][0], directions[t][1])
    #         ax.scatter(default_theta + math.pi, 1.5, marker='x')
    #
    #         plt.title(pos + " - " + str(t))
    #         plt.show()

    if show_plot == True:
        fs = summary.groupby(
            [summary.posture, summary.cursor_type, summary.wide]
        ).mean()

        fs.to_csv("total_basic_summary.csv")
        fs = pd.read_csv("total_basic_summary.csv")
        parameters = list(fs.columns)
        remove_columns = ["Unnamed: 0", "subject_num", "repetition", "target_num"]
        for removal in remove_columns:
            parameters.remove(removal)

        fig = px.bar(
            fs,
            x="cursor_type",
            y=parameters,
            barmode="group",
            facet_row="wide",
            facet_col="posture",
            title="total basic summary",
        )
        fig.show()
        fs_overall = summary.groupby([summary.posture, summary.cursor_type]).mean()
        # estimated target size
        fs_overall["estimated_width"] = (
            fs_overall.mean_offset_horizontal.apply(abs)
            + 2 * fs_overall.std_offset_horizontal
        )
        fs_overall["estimated_height"] = (
            fs_overall.mean_offset_vertical.apply(abs)
            + 2 * fs_overall.std_offset_vertical
        )
        fs_overall.to_csv("total_basic_summary_overall.csv")
        fs_overall = pd.read_csv("total_basic_summary_overall.csv")
        parameters.remove("wide")
        fig = px.bar(
            fs_overall,
            x="cursor_type",
            y=parameters + ["estimated_width", "estimated_height"],
            barmode="group",
            facet_col="posture",
            title="total basic summary",
        )
        fig.show()
        # wide = 20
        if show_distribution:
            for posture in ["WALK", "STAND"]:
                fig = go.Figure()
                plt_fig, plt_ax = plt.subplots()
                wide = 7.125
                x_offsets = [wide * math.sin(t * math.pi / 9 * 2) for t in range(9)]
                y_offsets = [wide * math.cos(t * math.pi / 9 * 2) for t in range(9)]
                plt_ax.scatter(x_offsets, y_offsets)
                wide = 14.04
                x_offsets = [wide * math.sin(t * math.pi / 9 * 2) for t in range(9)]
                y_offsets = [wide * math.cos(t * math.pi / 9 * 2) for t in range(9)]
                plt_ax.scatter(x_offsets, y_offsets)
                for t in range(9):
                    for idx, cursor_type in enumerate(["EYE", "HAND", "HEAD"]):
                        for w in ["LARGE", "SMALL"]:
                            if w == "LARGE":
                                wide = 14.04
                            else:
                                wide = 7.125
                            cursor_data = summary[
                                (summary["posture"] == posture)
                                & (summary["cursor_type"] == cursor_type)
                                & (summary["target_num"] == t)
                                & (summary["wide"] == w)
                            ]
                            cursor_data = cursor_data[cursor_data.error.isna() == True]
                            x_offset = wide * math.sin(t * math.pi / 9 * 2)
                            y_offset = wide * math.cos(t * math.pi / 9 * 2)
                            xs = cursor_data.mean_offset_horizontal + x_offset
                            ys = cursor_data.mean_offset_vertical + y_offset
                            color = DEFAULT_PLOTLY_COLORS[idx]
                            fig.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    name=cursor_type,
                                    mode="markers",
                                    marker={"color": color},
                                    opacity=0.1,
                                )
                            )
                            fig.add_shape(
                                type="path",
                                path=confidence_ellipse(xs, ys),
                                line={"dash": "dot"},
                                line_color=color,
                            )
                            colorset = [
                                "maroon",
                                "orangered",
                                "darkorange",
                                "olive",
                                "yellowgreen",
                                "darkolivegreen",
                                "turquoise",
                                "deepskyblue",
                                "dodgerblue",
                            ]
                            estimated_half_width = (
                                abs(cursor_data.mean_offset_horizontal.mean())
                                + 2 * cursor_data.std_offset_horizontal.mean()
                            )
                            estimated_half_height = (
                                abs(cursor_data.mean_offset_vertical.mean())
                                + 2 * cursor_data.std_offset_vertical.mean()
                            )
                            import matplotlib.patches as patches

                            plt_ax.add_patch(
                                patches.Rectangle(
                                    (
                                        x_offset - estimated_half_width,
                                        y_offset - estimated_half_height,
                                    ),
                                    2 * estimated_half_width,
                                    2 * estimated_half_height,
                                    edgecolor=colorset[t],
                                    fill=False,
                                )
                            )

                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                fig.add_hline(y=0)
                fig.add_vline(x=0)
                fig.show()
                plt.title(str(posture))
                plt.xlim(-30, 30)
                plt.ylim(-30, 30)
                plt.show()

    return summary


def discover_error(sub_num):
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]
    draw_plot = False
    cursorTypes = ["HEAD", "EYE", "HAND"]

    postures = ["WALK"]
    targets = range(9)
    # repetitions = [4, 5, 6, 7, 8, 9]
    repetitions = range(10)
    fail_count = 0
    # summary = pd.DataFrame(
    #     columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide',
    #              'error'])
    target_horizontal_acc = []
    walking_acc = []
    walking_speeds = []
    target_horizotal_vels = []

    for cursor_type in cursorTypes:
        for rep in repetitions:
            for pos in postures:
                # print('searching', sub_num, cursor_type, rep, pos)
                data = read_hololens_data(sub_num, pos, cursor_type, rep)
                splited_data = split_target(data)
                wide = "SMALL" if rep in rep_small else "LARGE"
                # for t in targets:
                try:

                    data["cursor_rotation"] = data.apply(
                        lambda x: asSpherical(
                            x.direction_x, x.direction_y, x.direction_z
                        ),
                        axis=1,
                    )
                    data["target_rotation"] = data.apply(
                        lambda x: asSpherical(
                            x.target_position_x - x.origin_x,
                            x.target_position_y - x.origin_y,
                            x.target_position_z - x.origin_z,
                        ),
                        axis=1,
                    )
                    data["cursor_horizontal"] = data.apply(
                        lambda x: math.sin(math.radians(x.cursor_rotation[1])), axis=1
                    )
                    data["cursor_vertical"] = data.apply(
                        lambda x: x.cursor_rotation[0], axis=1
                    )
                    data["target_horizontal"] = data.apply(
                        lambda x: math.sin(math.radians(x.target_rotation[1])), axis=1
                    )
                    data["target_vertical"] = data.apply(
                        lambda x: x.target_rotation[0], axis=1
                    )
                    data["horizontal"] = data.apply(
                        lambda x: change_angle(
                            x.cursor_horizontal - x.target_horizontal
                        ),
                        axis=1,
                    )
                    data["vertical"] = data.apply(
                        lambda x: change_angle(x.cursor_vertical - x.target_vertical),
                        axis=1,
                    )
                    data["target_vertical_velocity"] = (
                        (data["target_vertical"].diff(1) / data["timestamp"].diff(1))
                        .rolling(30)
                        .mean()
                    )
                    data["target_horizontal_velocity"] = data["target_horizontal"].diff(
                        1
                    ) / data["timestamp"].diff(1)
                    data["target_horizontal_acc"] = data[
                        "target_horizontal_velocity"
                    ].diff(1) / data["timestamp"].diff(1)
                    data["walking_speed"] = (
                        (
                            (
                                data["head_position_x"].diff(1).pow(2)
                                + data["head_position_z"].diff(1).pow(2)
                            ).apply(math.sqrt)
                            / data["timestamp"].diff(1)
                        )
                        .rolling(6, min_periods=1)
                        .mean()
                    )
                    data["walking_acc"] = data["walking_speed"].diff(1) / data[
                        "timestamp"
                    ].diff(1)
                    data["target_speed"] = (
                        (
                            (
                                data["target_position_x"].diff(1).pow(2)
                                + data["target_position_z"].diff(1).pow(2)
                            ).apply(math.sqrt)
                            / data["timestamp"].diff(1)
                        )
                        .rolling(6, min_periods=1)
                        .mean()
                    )
                    # target_horizontal_acc += list(data['target_horizontal_acc'])
                    # walking_acc += list(data['walking_acc'])
                    # walking_speeds += list(data['walking_speed'])
                    data["trial_check"] = data["end_num"].diff(1)
                    data_without_change = data[(data.trial_check == 0)][5:]
                    # plt.plot(data_without_change.timestamp, data_without_change.target_horizontal_velocity)
                    target_horizotal_vels += list(
                        data_without_change.target_horizontal_velocity
                    )
                    fail_check = data_without_change[
                        (data_without_change.target_horizontal_velocity > 5)
                        | (data_without_change.target_horizontal_velocity < -5)
                    ]
                    if len(fail_check) > 0:
                        fail_count += len(data.groupby(data.end_num).count())
                except Exception as e:
                    print(
                        sub_num, pos, cursor_type, rep, e.args, "fail count", fail_count
                    )
    # return target_horizontal_acc,walking_acc,walking_speeds
    return target_horizotal_vels, fail_count


def target_size_analysis(
    target_size,
    cursorTypes=None,
    postures=None,
    targets=range(9),
    repetitions=None,
    subjects=range(24),
):
    if postures is None:
        postures = ["STAND", "WALK"]
    if cursorTypes is None:
        cursorTypes = ["HEAD", "EYE", "HAND"]
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]
    draw_plot = False
    """ 
    Estimated tarrget size from accuracy/precision
    target width Sw = 2(Ox+26x)
    WALK
    Eye: width =  (abs(-0.091748219) + 2 * 2.321476905)
    HAND: width = (abs(0.276801368)+ 2 * 4.785825723)
    HEAD: width = (abs(0.037133211) + 2 * 2.564304494)
    """

    # repetitions = range(10)
    fail_count = 0
    summary = pd.DataFrame(
        columns=[
            "target_size",
            "subject_num",
            "posture",
            "cursor_type",
            "repetition",
            "target_num",
            "wide",
            "mean_offset",
            "std_offset",
            "overall_mean_offset",
            "overall_std_offset",
            "initial_contact_time",
            "target_in_count",
            "target_in_total_time",
            "target_in_mean_time",
            "mean_offset_horizontal",
            "mean_offset_vertical",
            "std_offset_horizontal",
            "std_offset_vertical",
            "longes t_dwell",
            "error",
        ]
    )
    for sub_num in subjects:
        for cursor_type in cursorTypes:
            for rep in repetitions:
                for pos in postures:
                    data = read_hololens_data(sub_num, pos, cursor_type, rep)
                    splited_data = split_target(data)
                    wide = "SMALL" if rep in rep_small else "LARGE"

                    for t in targets:
                        try:
                            trial_summary = {
                                "target_size": target_size,
                                "subject_num": sub_num,
                                "posture": pos,
                                "cursor_type": cursor_type,
                                "repetition": rep,
                                "target_num": t,
                                "wide": wide,
                            }
                            temp_data = splited_data[t]
                            temp_data.reset_index(inplace=True)
                            temp_data.timestamp -= temp_data.timestamp.values[0]
                            validate, reason = validate_trial_data(
                                temp_data, cursor_type, pos
                            )
                            if not validate:  # in case of invalid trial.
                                trial_summary["error"] = reason
                                summary.loc[len(summary)] = trial_summary
                                continue
                            drop_index = temp_data[
                                (temp_data["direction_x"] == 0)
                                & (temp_data["direction_y"] == 0)
                                & (temp_data["direction_z"] == 0)
                            ].index
                            # if len(drop_index) > 0:
                            #     print('drop length', len(drop_index), sub_num, pos, cursor_type, rep, t)
                            #     # raise ValueError
                            temp_data = temp_data.drop(drop_index)
                            # only_success = temp_data[temp_data.target_name == "Target_" + str(t)]
                            only_success = temp_data[
                                temp_data.cursor_angular_distance < target_size
                            ]

                            if len(only_success) <= 0:
                                raise ValueError("no success frames", len(only_success))
                            initial_contact_time = only_success.timestamp.values[0]

                            success_dwells = []
                            temp_data["target_in"] = (
                                temp_data["cursor_angular_distance"] < target_size
                            )
                            for k, g in itertools.groupby(
                                temp_data.iterrows(),
                                key=lambda row: row[1]["target_in"],
                            ):
                                # print(k, [t[0] for t in g])
                                if k == True:
                                    df = pd.DataFrame([r[1] for r in g])
                                    success_dwells.append(df)
                            time_sum = 0
                            times = []
                            for dw in success_dwells:
                                time_sum += (
                                    dw.timestamp.values[-1] - dw.timestamp.values[0]
                                )
                                times.append(
                                    dw.timestamp.values[-1] - dw.timestamp.values[0]
                                )
                            longest_dwell = max(times)
                            target_in_count = len(success_dwells)
                            target_in_total_time = time_sum
                            target_in_mean_time = time_sum / target_in_count

                            dwell_temp = temp_data[
                                temp_data.timestamp >= initial_contact_time
                            ]
                            #
                            mean_offset_horizontal = dwell_temp.horizontal_offset.mean()
                            std_offset_horizontal = dwell_temp.horizontal_offset.std()
                            mean_offset_vertical = dwell_temp.vertical_offset.mean()
                            std_offset_vertical = dwell_temp.vertical_offset.std()

                            # dwell threshold and target-size analysis
                            # dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                            target_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

                            # for threshold in dwell_thresholds:
                            #     for target_size in target_sizes:
                            #         # in each estimation
                            #         first_dwell_time = None
                            #         dwell_count = 0
                            #         dwell_total_hover_time = 0
                            #         dwell_mean_hover_time = 0
                            #         # gets the target-in frames depends on different target size
                            #         temp_data['temp_dwell_check'] = temp_data[
                            #                                             'cursor_angular_distance'] < target_size
                            #         temp_success_dwells = []
                            #         for k, g in itertools.groupby(temp_data.iterrows(),
                            #                                       key=lambda row: row[1]['temp_dwell_check']):
                            #             if k == True:
                            #                 df = pd.DataFrame([r[1] for r in g])
                            #                 temp_success_dwells.append(df)
                            #         if len(temp_success_dwells) <= 0:  # if there is no successful dwell
                            #             continue
                            #         for index, dw in enumerate(temp_success_dwells):
                            #             if len(dw.values) <= 1:
                            #                 continue
                            #             if index == 0:
                            #                 first_dwell_time = dw.timestamp.values[0]
                            #             current_dwell_time = dw.timestamp.values[-1] - dw.timestamp.values[0]
                            #             if current_dwell_time >= threshold:  # successful dwell
                            #                 dwell_total_hover_time += current_dwell_time
                            #                 dwell_count += 1

                            trial_summary = {
                                "target_size": target_size,
                                "subject_num": sub_num,
                                "posture": pos,
                                "cursor_type": cursor_type,
                                "repetition": rep,
                                "target_num": t,
                                "wide": wide,
                                "mean_offset": dwell_temp.cursor_angular_distance.mean(),
                                "std_offset": dwell_temp.cursor_angular_distance.std(),
                                "overall_mean_offset": temp_data.cursor_angular_distance.mean(),
                                "overall_std_offset": temp_data.cursor_angular_distance.mean(),
                                "initial_contact_time": initial_contact_time,
                                "target_in_count": float(target_in_count),
                                "target_in_total_time": target_in_total_time,
                                "target_in_mean_time": target_in_mean_time,
                                "mean_offset_horizontal": mean_offset_horizontal,
                                "mean_offset_vertical": mean_offset_vertical,
                                "std_offset_horizontal": std_offset_horizontal,
                                "std_offset_vertical": std_offset_vertical,
                                "longest_dwell": longest_dwell,
                                "error": None,
                            }
                            summary.loc[len(summary)] = trial_summary
                            # smalls.loc[len(smalls)] = [sub_num, 'STAND', cursor_type, rep, t, temp_data.cursor_angular_distance.mean(),
                            #                            temp_data.cursor_angular_distance.std()]
                        except Exception as e:
                            fail_count += 1
                            error_summary = {
                                "target_size": target_size,
                                "subject_num": sub_num,
                                "posture": pos,
                                "cursor_type": cursor_type,
                                "repetition": rep,
                                "target_num": t,
                                "wide": wide,
                                # 'mean_offset': None,
                                # 'std_offset': None,
                                # 'initial_contact_time': None,
                                # 'target_in_count': None,
                                # 'target_in_total_time': None,
                                # 'target_in_mean_time': None,
                                # 'mean_offset_horizontal': None,
                                # 'mean_offset_vertical': None,
                                # 'std_offset_horizontal': None,
                                # 'std_offset_vertical': None,
                                "error": e.args,
                            }
                            summary.loc[len(summary)] = error_summary
                            print(
                                sub_num,
                                pos,
                                cursor_type,
                                rep,
                                t,
                                e.args,
                                "fail count",
                                fail_count,
                            )
                            # plt.plot(temp_data.timestamp, temp_data.cursor_angular_distance, label=str(pos))
                            # # plt.plot(walk_temp.timestamp, walk_temp.cursor_angular_distance, label='walk')
                            # plt.axvline(initial_contact_time)
                            # # plt.axvline(walk_initial_contact_time)
                            # # plt.text(initial_contact_time,0,'initial contact time',)
                            # plt.legend()
                            # plt.title(f'{cursor_type} angular offset from :target' + str(t))
                            # plt.show()
    final_summary = summary.groupby(
        [summary["posture"], summary["cursor_type"], summary["wide"]]
    ).mean()
    final_summary.to_csv("target_size_summary" + str(target_size) + ".csv")
    summary.to_csv("target_size_Rawsummary" + str(target_size) + ".csv")
    return summary


@timeit
def newStudy_basic_analysis(data, dt=1.0):
    data.timestamp -= data.timestamp.values[0]
    endnum = data.end_num.values[0]
    data["success"] = data.target_name == "Target_" + str(endnum)
    only_success = data[data["success"] == True]
    if len(only_success) <= 0:
        print("no success")
        return data, None
    initial_contact_time = only_success.timestamp.values[0]
    dwell_temp = data[data.timestamp >= initial_contact_time]
    mean_error = dwell_temp.cursor_angular_distance.mean()
    median_error = dwell_temp.cursor_angular_distance.median()
    mrt_df = data[
        (data.timestamp >= initial_contact_time)
        & (data.timestamp <= dt + initial_contact_time)
    ]
    mrt = max(list(mrt_df.cursor_angular_distance))
    all_success_dwells = []

    # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_in']):
    for k, g in itertools.groupby(data.iterrows(), key=lambda row: row[1]["success"]):
        # print(k, [t[0] for t in g])
        # if k == True:
        if k == True:
            # if k == 'Target_' + str(t):
            df = pd.DataFrame([r[1] for r in g])
            if len(df) <= 1:
                continue
            all_success_dwells.append(df)

    success_dwells = []
    times = []
    dwell_time = dt
    for dw in all_success_dwells:
        time_record = dw.timestamp.values[-1] - dw.timestamp.values[0]
        times.append(time_record)
        if time_record >= dwell_time:
            success_dwells.append(dw)
    best_record = max(times)
    time_sum = 0
    for dw in success_dwells:
        record = dw.timestamp.values[-1] - dw.timestamp.values[0]
        time_sum += record

    trial_duration = success_dwells[0].timestamp.values[-1]
    target_in_count = 0
    for a in all_success_dwells:
        if a.timestamp.values[0] < trial_duration:
            target_in_count = target_in_count + 1
    first_dwell_time = None
    for index, dw in enumerate(success_dwells):
        if index == 0:
            first_dwell_time = dw.timestamp.values[0]


def rotate_towards_prev(
    current_vector, target_vector, max_angle_degrees, max_magnitude_change
):
    # Normalize the input vectors
    current_vector = np.array(current_vector) / np.linalg.norm(current_vector)
    target_vector = np.array(target_vector) / np.linalg.norm(target_vector)

    dot_product = np.dot(current_vector, target_vector)
    angle_between = np.deg2rad(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    angle = min(np.deg2rad(max_angle_degrees), angle_between)

    axis_of_rotation = np.cross(current_vector, target_vector)
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    rotated_vector = np.dot(rotation_matrix, current_vector)
    # return rotated_vector
    return (
        rotated_vector * (1 - max_magnitude_change)
        + target_vector * max_magnitude_change
    )


def move_towards(
    current: np.ndarray, target: np.ndarray, max_distance: float
) -> np.ndarray:
    """
    Moves a point (current) towards a target point by a maximum distance.

    Args:
        current: The current position (2D NumPy array of floats).
        target: The target position (2D NumPy array of floats).
        max_distance: The maximum distance to move in this step (float).

    Returns:
        The new position after the move (2D NumPy array of floats).
    """
    # Calculate the direction vector
    direction = target - current

    # Get the magnitude of the direction vector
    distance = np.linalg.norm(direction)

    # Clamp the distance to the max allowed movement
    move_amount = min(distance, max_distance)

    # Normalize the direction vector
    normalized_direction = direction / (distance or 1e-6)  # Avoid division by zero

    # Move along the direction by the allowed amount
    new_position = current + move_amount * normalized_direction

    return new_position


@timeit
def target_expansion_simulation_subject(subs, rate,width=None,height=None):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "speed_limit",
            "bound" "overall_time",
            "success",
            "target_in_count",
            "first_dwell_time",
            "error",
        ]
    )
    subjects = subs
    repetitions = range(1, 5)
    cursorTypes = ["Head", "Eye", "Hand"]
    postures = ["Stand", "Treadmill", "Walk"]
    selections = ["Dwell", "Click"]
    summaries = []
    for sub_num, cursor_type, sel, rep, pos in itertools.product(
        subjects, cursorTypes, selections, repetitions, postures
    ):
        one_file_summary = target_expansion_simulation(
            sub_num, sel, cursor_type, pos, rep, rate,width,height
        )
        summaries.append(one_file_summary)
        # summary.loc[len(summary)] = one_trial_summary
    return pd.concat(summaries)


@timeit
def speed_reduction_simulation_subject(subs, speed_limit, bound):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "speed_limit",
            "bound" "overall_time",
            "success",
            "target_in_count",
            "first_dwell_time",
            "error",
        ]
    )
    subjects = subs
    repetitions = range(1, 5)
    cursorTypes = ["Head", "Eye", "Hand"]
    postures = ["Treadmill", "Walk"]
    selections = ["Click", "Dwell"]
    summaries = []
    for sub_num, cursor_type, sel, rep, pos in itertools.product(
        subjects, cursorTypes, selections, repetitions, postures
    ):
        one_file_summary = speed_reduction_simulation(
            sub_num, sel, cursor_type, pos, rep, speed_limit, bound
        )
        summaries.append(one_file_summary)
        # summary.loc[len(summary)] = one_trial_summary
    return pd.concat(summaries)


def map_range(value, start, end):
    if value < start:
        return start
    elif value >= end:
        return end
    else:
        return value


def target_expansion_simulation(sub, sel, cur, pos, r, rate,width,height):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "std_error",
            "best_record",
            "speed_limit",
            "bound",
            "rate",
            "overall_time",
            "success",
        ]
    )
    try:
        d, success_record = read_data(sub, r, cur, sel, pos)
        data = split_target(d)
    except Exception as e:
        print(e, sub, r, cur, sel, pos)
        return None
    for t in range(9):
        try:
            trial_summary = {
                "subject": sub,
                "posture": pos,
                "cursor": cur,
                "selection": sel,
                "target": t,
                "repetition": r,
                "rate": rate,
            }
            temp_data = data[t].reset_index()
            temp_data.timestamp -= temp_data.timestamp.values[0]
            # temp_data['EH_distance'] = ((temp_data.head_horizontal_offset - temp_data.eyeRay_horizontal_offset) ** 2 + (
            #         temp_data.head_vertical_offset - temp_data.eyeRay_vertical_offset) ** 2).apply(math.sqrt)
            # temp_data['head_direction_vector'] = temp_data.apply(
            #     lambda row: directional_vector(row['head_horizontal_offset'], row['head_vertical_offset']), axis=1)
            # temp_data['eye_direction_vector'] = temp_data.apply(
            #     lambda row: directional_vector(row['eyeRay_horizontal_offset'], row['eyeRay_vertical_offset']), axis=1)
            # temp_data['newcursor_direction_vector'] = temp_data['head_direction_vector']
            temp_data["o_success"] = temp_data.cursor_angular_distance <= 3.0
            only_success = temp_data[
                (temp_data.o_success == True)
                | (temp_data.target_name == "Target_" + str(t))
            ]
            if len(only_success) <= 0:
                print("no touch in ", sub, sel, cur, pos, r, t)
                trial_summary["error"] = "no touch"
                summary.loc[len(summary)] = trial_summary
                continue

            initial_contact_time = only_success.timestamp.values[0]
            if width !=None :
                temp_data["expansion_width"] = (
                    (temp_data.timestamp - initial_contact_time) * rate
                ).apply(map_range, start=0, end=width-3) + 3
                temp_data["expansion_height"] = (
                    (temp_data.timestamp - initial_contact_time) * rate
                ).apply(map_range, start=0, end=height-3) + 3
                
                temp_data["success"] = temp_data.apply(
                    lambda x: (
                        (x["horizontal_offset"] ** 2) / (x["expansion_width"] * x["expansion_width"])
                + (x["vertical_offset"] ** 2) / (x["expansion_height"] * x["expansion_height"])
                <= 1
                  )
                    | (x.target_name == "Target_" + str(t)),
                    axis=1,
                )
            else:
                temp_data["expansion"] = (
                    (temp_data.timestamp - initial_contact_time) * rate
                ).apply(map_range, start=0, end=3) + 3
            # after_contact = temp_data[temp_data.timestamp >= initial_contact_time]
            # mean_offset = after_contact.newcursor_angular_distance.mean()
            # std_offset = after_contact.newcursor_angular_distance.std()
                
                temp_data["success"] = temp_data.apply(
                    lambda x: (x.cursor_angular_distance <= x.expansion)
                    | (x.target_name == "Target_" + str(t)),
                    axis=1,
                )
            # temp_data['success'] = temp_data[(temp_data.cursor_angular_distance <= temp_data.expansion)]
            # | (
            #         temp_data.target_name == "Target_" + str(t))
            all_success_dwells = []
            for k, g in itertools.groupby(
                temp_data.iterrows(), key=lambda row: row[1]["success"]
            ):
                if k == True:
                    df = pd.DataFrame([r[1] for r in g])
                    all_success_dwells.append(df)
            success_dwells = []
            times = []
            # target_in_count = len(all_success_dwells)
            for dw in all_success_dwells:
                time_record = dw.timestamp.values[-1] - dw.timestamp.values[0]
                times.append(time_record)
                if time_record >= 1.0 - 1.5 / 60:
                    success_dwells.append(dw)
            if sel == "Dwell":
                if len(success_dwells) > 0:
                    success = True
                else:
                    success = False
            else:
                if temp_data["success"].values[-1] == True:
                    success = True
                else:
                    success = False
            if success_record[t] == "O":
                success = True
            trial_summary["initial_contact_time"] = initial_contact_time
            # trial_summary['mean_error'] = mean_offset
            # trial_summary['std_error'] = std_offset
            # trial_summary['target_in_count'] = target_in_count
            trial_summary["success"] = success
            summary.loc[len(summary)] = trial_summary
        except Exception as e:
            pass
            # print(e.args, sub, r, cur, sel, pos)
    return summary


def speed_reduction_simulation(sub, sel, cur, pos, r, speed_limit, bound):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "std_error",
            "best_record",
            "speed_limit",
            "bound",
            "overall_time",
            "success",
            "target_in_count",
            "target_in_count_per_second",
            "walk_speed",
            "first_dwell_time",
            "error",
            "final_point_horizontal",
            "final_point_vertical",
            "final_cursor_speed",
        ]
    )
    try:
        d, success_record = read_data(sub, r, cur, sel, pos)
        data = split_target(d)
    except Exception as e:
        print(e, sub, r, cur, sel, pos)
        return None

    for t in range(9):
        try:
            trial_summary = {
                "subject": sub,
                "posture": pos,
                "cursor": cur,
                "selection": sel,
                "target": t,
                "repetition": r,
                "speed_limit": speed_limit,
                "bound": bound,
            }
            temp_data = data[t].reset_index()
            temp_data.timestamp -= temp_data.timestamp.values[0]
            temp_data["EH_distance"] = (
                (temp_data.head_horizontal_offset - temp_data.eyeRay_horizontal_offset)
                ** 2
                + (temp_data.head_vertical_offset - temp_data.eyeRay_vertical_offset)
                ** 2
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
            # temp_data['newcursor_direction_vector'] = temp_data['head_direction_vector']

            temp_data["cursor_direction"] = temp_data.apply(
                lambda row: [row.direction_x, row.direction_y, row.direction_z], axis=1
            )
            # temp_data['newcursor_direction_vector'] = temp_data['cursor_direction']
            temp_data["newcursor_horizontal_angle"] = temp_data["horizontal_offset"]
            temp_data["newcursor_vertical_angle"] = temp_data["vertical_offset"]
            temp_data["success"] = temp_data.cursor_angular_distance <= 3.0
            only_success = temp_data[
                (temp_data.success == True)
                | (temp_data.target_name == "Target_" + str(t))
            ]
            if len(only_success) <= 0:
                print("no touch in ", sub, sel, cur, pos, r, t)
                # trial_summary['error'] = "no touch"
                # summary.loc[len(summary)] = trial_summary
                continue

            initial_contact_time = only_success.timestamp.values[0]
            for index, row in temp_data.iterrows():
                if row["timestamp"] < initial_contact_time:
                    continue
                eh = temp_data["EH_distance"][index]
                # speed = speed_limit if eh < bound else 100
                speed = speed_limit

                newcursor_position_ = move_towards(
                    np.array(
                        [
                            temp_data.newcursor_horizontal_angle[index - 1],
                            temp_data.newcursor_vertical_angle[index - 1],
                        ]
                    ),
                    np.array(
                        [
                            temp_data.horizontal_offset[index],
                            temp_data.vertical_offset[index],
                        ]
                    ),
                    (speed * (1 / 60)),
                )
                temp_data["newcursor_horizontal_angle"][index] = newcursor_position_[0]
                temp_data["newcursor_vertical_angle"][index] = newcursor_position_[1]

            # temp_data['newcursor_direction_vector_x'] = temp_data.apply(lambda x: x.newcursor_direction_vector[0],
            #                                                             axis=1)
            # temp_data['newcursor_direction_vector_y'] = temp_data.apply(lambda x: x.newcursor_direction_vector[1],
            #                                                             axis=1)
            # temp_data['newcursor_direction_vector_z'] = temp_data.apply(lambda x: x.newcursor_direction_vector[2],
            #                                                             axis=1)
            # temp_data['newcursor_rotation'] = temp_data.apply(
            #     lambda x: asSpherical(x.newcursor_direction_vector_x, x.newcursor_direction_vector_y,
            #                           x.newcursor_direction_vector_z), axis=1)
            # temp_data['newcursor_horizontal_angle'] = temp_data.apply(
            #     lambda x: x.newcursor_position_vector[0], axis=1
            # )
            # temp_data['newcursor_vertical_angle'] = temp_data.apply(
            #     lambda x: x.newcursor_position_vector[1], axis=1
            # )
            # temp_data['newcursor_vertical_angle'] = 90 - temp_data['newcursor_vertical_angle']
            temp_data["newcursor_angular_distance"] = (
                temp_data["newcursor_horizontal_angle"] ** 2
                + temp_data["newcursor_vertical_angle"] ** 2
            ).apply(math.sqrt)

            # initial_contact_time = temp_data[temp_data['newcursor_angular_distance'] <= 3.0].timestamp.values[0]
            # after_contact = temp_data[temp_data.timestamp >= initial_contact_time]
            # mean_offset = after_contact.newcursor_angular_distance.mean()
            # std_offset = after_contact.newcursor_angular_distance.std()
            all_success_dwells = []
            # temp_data['success'] = temp_data.newcursor_angular_distance <= 3.0
            temp_data.loc[
                (temp_data["newcursor_angular_distance"] <= 3.0)
                | (temp_data.target_name == "Target_" + str(t)),
                "success",
            ] = True

            # temp_data[(temp_data.success == True) | (temp_data.target_name == "Target_" + str(t))]
            if sel == "Dwell":
                for k, g in itertools.groupby(
                    temp_data.iterrows(), key=lambda row: row[1]["success"]
                ):
                    if k == True:
                        df = pd.DataFrame([r[1] for r in g])
                        all_success_dwells.append(df)
                success_dwells = []
                times = []
                target_in_count = len(all_success_dwells)
                for dw in all_success_dwells:
                    time_record = dw.timestamp.values[-1] - dw.timestamp.values[0]
                    times.append(time_record)
                    if time_record >= 1.0 - 1.5 / 60:
                        success_dwells.append(dw)
                if len(success_dwells) > 0:
                    success = True
                else:
                    success = False
            else:
                if temp_data.success.values[-1]:
                    success = True
                else:
                    success = False
            if success_record[t] == "O":
                success = True
            # trial_summary['initial_contact_time'] = initial_contact_time
            # trial_summary['mean_error'] = mean_offset
            # trial_summary['std_error'] = std_offset
            # trial_summary['target_in_count'] = target_in_count
            trial_summary["success"] = success
            summary.loc[len(summary)] = trial_summary
        except Exception:
            # pass
            print(traceback.format_exc(), sub, r, cur, sel, pos, t)
    return summary


def directional_vector(horizontal_angle_degrees, vertical_angle_degrees):
    # Convert angles to radians
    horizontal_angle_rad = np.deg2rad(horizontal_angle_degrees)
    vertical_angle_rad = np.deg2rad(vertical_angle_degrees)

    # Calculate the directional vector components
    x_component = np.cos(horizontal_angle_rad) * np.cos(vertical_angle_rad)
    y_component = np.sin(vertical_angle_rad)
    z_component = np.sin(horizontal_angle_rad) * np.cos(vertical_angle_rad)

    return [z_component, y_component, x_component]


def normalize_vector(vector):
    # Convert the input vector to a NumPy array
    vector = np.array(vector)

    # Calculate the magnitude of the vector
    magnitude = np.linalg.norm(vector)

    # Check if the magnitude is not zero (to avoid division by zero)
    if magnitude != 0:
        # Normalize the vector by dividing each component by its magnitude
        normalized_vector = vector / magnitude
        return normalized_vector
    else:
        # If the magnitude is zero, return the original vector (or handle as needed)
        return vector


@timeit
def threshold_analysis_subject(subs, threshold=None):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "overall_time",
            "success",
            "target_in_count",
            "first_dwell_time",
            "error",
        ]
    )
    subjects = subs
    cursorTypes = ["Head", "Eye", "Hand"]
    repetitions = range(1, 5)
    postures = ["Stand", "Treadmill", "Walk"]
    selections = ["Dwell"]
    summaries = []
    for sub_num, cursor_type, sel, rep, pos in itertools.product(
        subjects, cursorTypes, selections, repetitions, postures
    ):
        one_file_summary = dwell_threshold_file(
            sub_num, sel, cursor_type, pos, rep, threshold=threshold
        )
        summaries.append(one_file_summary)
        # summary.loc[len(summary)] = one_trial_summary
    return pd.concat(summaries)


@timeit
def basic_analysis_subject(subs, threshold=None, speed=False):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "overall_time",
            "success",
            "target_in_count",
            "first_dwell_time",
            "error",
        ]
    )
    subjects = subs
    cursorTypes = ["Head", "Eye", "Hand"]
    repetitions = range(1, 5)
    postures = ["Stand", "Treadmill", "Walk"]
    selections = ["Click", "Dwell"]
    # selections = ['Dwell']
    # targets = range(9)
    summaries = []
    for sub_num, cursor_type, sel, rep, pos in itertools.product(
        subjects, cursorTypes, selections, repetitions, postures
    ):
        one_file_summary = basic_analysis_file(
            sub_num, sel, cursor_type, pos, rep, threshold=threshold, speed=speed
        )
        summaries.append(one_file_summary)
        # summary.loc[len(summary)] = one_trial_summary
    return pd.concat(summaries)


@timeit
def Target_size_basic_analysis_subject(subs, _width, _height, threshold=None):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "overall_time",
            "success",
            "target_in_count",
            "first_dwell_time",
            "error",
        ]
    )
    subjects = subs
    cursorTypes = ["Head", "Eye", "Hand"]
    repetitions = range(1, 5)
    postures = ["Stand", "Walk", "Treadmill"]
    selections = ["Dwell", "Click"]
    # selections = ['Dwell']
    # targets = range(9)
    summaries = []
    for sub_num, cursor_type, sel, rep, pos in itertools.product(
        subjects, cursorTypes, selections, repetitions, postures
    ):
        one_file_summary = Target_size_basic_analysis_file(
            sub_num,
            sel,
            cursor_type,
            pos,
            rep,
            _width=_width,
            _height=_height,
            threshold=threshold,
        )
        summaries.append(one_file_summary)
        # summary.loc[len(summary)] = one_trial_summary
    return pd.concat(summaries)


def dwell_threshold_file(sub, sel, cur, pos, r, threshold):  # dwell only
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "overall_time",
            "success",
            "target_in_count",
            "target_in_count_per_second",
            "walk_speed",
            "first_dwell_time",
            "error",
            "final_point_horizontal",
            "final_point_vertical",
            "final_cursor_speed",
        ]
    )
    try:
        d, success_record = read_data(sub, r, cur, sel, pos)
        data = split_target(d)
    except Exception as e:
        print(e, sub, r, cur, sel, pos)
        return None

    for t in range(9):
        try:
            trial_summary = {
                "subject": sub,
                "posture": pos,
                "cursor": cur,
                "selection": sel,
                "target": t,
                "repetition": r,
                "threshold": threshold,
            }
            temp_data = data[t]
            temp_data.timestamp -= temp_data.timestamp.values[0]
            temp_data["success"] = temp_data.cursor_angular_distance <= 3.0
            only_success = temp_data[
                (temp_data.success == True)
                | (temp_data.target_name == "Target_" + str(t))
            ]
            if len(only_success) <= 0:
                print("no touch in ", sub, sel, cur, pos, r, t)
                trial_summary["error"] = "no touch"
                summary.loc[len(summary)] = trial_summary
                continue

            initial_contact_time = only_success.timestamp.values[0]
            tt = temp_data[temp_data.timestamp >= initial_contact_time]
            mean_error = tt.cursor_angular_distance.mean()
            trial_summary["mean_error"] = mean_error
            all_success_dwells = []
            for k, g in itertools.groupby(
                temp_data.iterrows(), key=lambda row: row[1]["success"]
            ):
                if k == True:
                    df = pd.DataFrame([r[1] for r in g])
                    all_success_dwells.append(df)
            success_dwells = []
            times = []
            for dw in all_success_dwells:
                time_record = dw.timestamp.values[-1] - dw.timestamp.values[0]
                times.append(time_record)
                if time_record >= threshold - 2.5 / 60:
                    success_dwells.append(dw)
            if len(success_dwells) > 0:
                trial_summary["success"] = True
                endtime = success_dwells[0].timestamp.values[-1]
                starttime = success_dwells[0].timestamp.values[0]
                dwell_temp = temp_data[
                    (temp_data.timestamp >= starttime)
                    & (temp_data.timestamp <= endtime)
                ]
                trial_summary["mean_error_horizontal"] = (
                    dwell_temp.horizontal_offset.mean()
                )
                trial_summary["mean_error_vertical"] = dwell_temp.vertical_offset.mean()
                trial_summary["std_error_horizontal"] = (
                    dwell_temp.horizontal_offset.std()
                )
                trial_summary["std_error_vertical"] = dwell_temp.vertical_offset.std()
                trial_summary["mean_cursor_speed"] = (
                    abs(dwell_temp.cursor_angular_distance.diff())
                    / dwell_temp.timestamp.diff()
                ).mean()

                trial_summary["final_cursor_speed"] = (
                    abs(dwell_temp.cursor_angular_distance.diff())
                    / dwell_temp.timestamp.diff()
                )[-6:].mean()
                trial_summary["initial_contact_time"] = initial_contact_time
                trial_summary["first_dwell_time"] = success_dwells[0].timestamp.values[
                    0
                ]
                trial_summary["endtime"] = endtime
                trial_summary["starttime"] = starttime
                target_in_count = 0
                for dw in all_success_dwells:
                    if dw.timestamp.values[0] < endtime:
                        target_in_count = target_in_count + 1
                trial_summary["target_in_count"] = target_in_count
                trial_summary["target_in_count_per_second"] = target_in_count / (
                    dwell_temp.timestamp.values[-1] - dwell_temp.timestamp.values[0]
                )
                summary.loc[len(summary)] = trial_summary
                continue
            else:
                if success_record[t] == "O":
                    trial_summary["success"] = True
                else:
                    trial_summary["success"] = False
                summary.loc[len(summary)] = trial_summary
                continue
        except Exception as e:
            print(e, sub, r, cur, sel, pos)
    return summary


def basic_analysis_file(sub, sel, cur, pos, r, threshold=None, speed=False):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "overall_time",
            "success",
            "target_in_count",
            "target_in_count_per_second",
            "walk_speed",
            "first_dwell_time",
            "error",
            "final_point_horizontal",
            "final_point_vertical",
            "final_cursor_speed",
            "dwell_speeds",
            "before_speeds",
        ]
    )
    try:
        d, success_record = read_data(sub, r, cur, sel, pos)
        data = split_target(d)
    except Exception as e:
        print(e, sub, r, cur, sel, pos)
        return None

    for t in range(9):
        try:
            trial_summary = {
                "subject": sub,
                "posture": pos,
                "cursor": cur,
                "selection": sel,
                "target": t,
                "repetition": r,
            }

            if success_record[t] == "O":
                success = True
            else:
                success = False
            dwell_time = 1.0
            if threshold != None:
                dwell_time = threshold
            # trial_summary['threshold'] = dwell_time
            trial_summary["success"] = success
            temp_data = data[t]
            temp_data.timestamp -= temp_data.timestamp.values[0]
            overall_time = temp_data.timestamp.values[-1]
            trial_summary["overall_time"] = overall_time
            temp_data["success"] = temp_data.cursor_angular_distance <= 3.0
            # temp_data['success'] = temp_data.target_name == "Target_" + str(t)
            only_success = temp_data[
                (temp_data.success == True)
                | (temp_data.target_name == "Target_" + str(t))
            ]

            if sel == "Dwell":
                if len(only_success) <= 0:
                    print("no touch in ", sub, sel, cur, pos, r, t)
                    trial_summary["error"] = "no touch"
                    summary.loc[len(summary)] = trial_summary
                    continue
                initial_contact_time = only_success.timestamp.values[0]
                walklength = (
                    (
                        temp_data.head_origin_x.diff(1) ** 2
                        + temp_data.head_origin_y.diff(1) ** 2
                        + temp_data.head_origin_z.diff(1) ** 2
                    )
                    .apply(math.sqrt)
                    .sum()
                )
                walk_speed = walklength / (overall_time - initial_contact_time)
                trial_summary["walk_speed"] = walk_speed
                dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                before_temp = temp_data[temp_data.timestamp < initial_contact_time]
                mean_error = dwell_temp.cursor_angular_distance.mean()
                trial_summary["mean_error"] = mean_error
                trial_summary["mean_error_horizontal"] = (
                    dwell_temp.horizontal_offset.mean()
                )
                trial_summary["mean_error_vertical"] = dwell_temp.vertical_offset.mean()
                trial_summary["std_error_horizontal"] = (
                    dwell_temp.horizontal_offset.std()
                )
                trial_summary["std_error_vertical"] = dwell_temp.vertical_offset.std()
                trial_summary["mean_cursor_speed"] = (
                    abs(dwell_temp.cursor_angular_distance.diff())
                    / dwell_temp.timestamp.diff()
                ).mean()
                if speed:
                    trial_summary["dwell_speeds"] = (
                        dwell_temp.cursor_angular_distance.diff()
                    )
                    trial_summary["before_speeds"] = (
                        before_temp.cursor_angular_distance.diff()
                    )
                mrt_df = temp_data[
                    (temp_data.timestamp >= initial_contact_time)
                    & (temp_data.timestamp <= dwell_time + initial_contact_time)
                ]
                mrt = max(list(mrt_df.cursor_angular_distance))
                trial_summary["initial_contact_time"] = initial_contact_time
                trial_summary["required_target_size"] = mrt
                all_success_dwells = []
                for k, g in itertools.groupby(
                    temp_data.iterrows(), key=lambda row: row[1]["success"]
                ):
                    if k == True:
                        df = pd.DataFrame([r[1] for r in g])
                        all_success_dwells.append(df)
                success_dwells = []
                times = []

                target_in_count = len(all_success_dwells)
                trial_summary["target_in_count"] = target_in_count
                target_in_count_per_second = target_in_count / (
                    overall_time - initial_contact_time
                )
                trial_summary["target_in_count_per_second"] = target_in_count_per_second
                for dw in all_success_dwells:
                    time_record = dw.timestamp.values[-1] - dw.timestamp.values[0]
                    times.append(time_record)
                    if time_record >= dwell_time - 1.5 / 60:
                        success_dwells.append(dw)
                if success:
                    trial_summary["first_dwell_time"] = overall_time - 1 + 1.5 / 60
                # if len(success_dwells) >0:
                # trial_summary['']
                summary.loc[len(summary)] = trial_summary
                continue
            elif sel == "Click":
                trial_summary["final_point_horizontal"] = (
                    temp_data.horizontal_offset.values[-1]
                )
                trial_summary["final_point_vertical"] = (
                    temp_data.vertical_offset.values[-1]
                )
                if len(only_success) <= 0:
                    # print('no touch in ', sub, sel, cur, pos, r, t)
                    trial_summary["error"] = "no touch"
                    summary.loc[len(summary)] = trial_summary
                    continue
                initial_contact_time = only_success.timestamp.values[0]
                trial_summary["initial_contact_time"] = initial_contact_time
                trial_summary["final_cursor_speed"] = abs(
                    temp_data.cursor_angular_distance.diff()
                    / temp_data.timestamp.diff()
                )[-12:].mean()
                walklength = (
                    (
                        temp_data.head_origin_x.diff(1) ** 2
                        + temp_data.head_origin_y.diff(1) ** 2
                        + temp_data.head_origin_z.diff(1) ** 2
                    )
                    .apply(math.sqrt)
                    .sum()
                )
                walk_speed = walklength / (overall_time - initial_contact_time)
                trial_summary["walk_speed"] = walk_speed

                all_success_dwells = []
                for k, g in itertools.groupby(
                    temp_data.iterrows(), key=lambda row: row[1]["success"]
                ):
                    if k == True:
                        df = pd.DataFrame([r[1] for r in g])
                        all_success_dwells.append(df)
                success_dwells = []
                times = []

                target_in_count = len(all_success_dwells)
                if target_in_count > 0:
                    trial_summary["target_in_count"] = target_in_count
                    target_in_count_per_second = target_in_count / (
                        overall_time - initial_contact_time
                    )
                    trial_summary["target_in_count_per_second"] = (
                        target_in_count_per_second
                    )
                    dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                    trial_summary["mean_cursor_speed"] = (
                        abs(dwell_temp.cursor_angular_distance.diff())
                        / dwell_temp.timestamp.diff()
                    ).mean()

                summary.loc[len(summary)] = trial_summary
                continue
        except Exception as e:
            print(e, sub, r, cur, sel, pos)
    return summary


def Target_size_basic_analysis_file(
    sub,
    sel,
    cur,
    pos,
    r,
    _width,
    _height,
    threshold=None,
):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "initial_contact_time",
            "mean_error",
            "best_record",
            "threshold",
            "required_target_size",
            "mean_error_horizontal",
            "mean_error_vertical",
            "std_error_horizontal",
            "std_error_vertical",
            "mean_cursor_speed",
            "overall_time",
            "success",
            "target_in_count",
            "target_in_count_per_second",
            "walk_speed",
            "first_dwell_time",
            "error",
            "final_point_horizontal",
            "final_point_vertical",
            "final_cursor_speed",
        ]
    )
    try:
        d, success_record = read_data(sub, r, cur, sel, pos)
        data = split_target(d)
    except Exception as e:
        print(e, sub, r, cur, sel, pos)
        return None

    for t in range(9):
        try:
            trial_summary = {
                "subject": sub,
                "posture": pos,
                "cursor": cur,
                "selection": sel,
                "target": t,
                "repetition": r,
            }

            if success_record[t] == "O":
                success = True
            else:
                success = False
            dwell_time = 1.0
            if threshold != None:
                dwell_time = threshold
            # trial_summary['threshold'] = dwell_time
            # trial_summary['success'] = success
            temp_data = data[t]
            temp_data.timestamp -= temp_data.timestamp.values[0]
            overall_time = temp_data.timestamp.values[-1]
            trial_summary["overall_time"] = overall_time
            # temp_data['success'] = (temp_data.horizontal_offset.apply(abs) < _width) and (
            #             temp_data.vertical_offset.apply(abs) < _height)

            # temp_data.loc[(temp_data["horizontal_offset"].apply(abs) <= _width) & (
            #         temp_data["vertical_offset"].apply(abs) < _height), "success"] = True
            temp_data.loc[
                (temp_data["horizontal_offset"] ** 2) / (_width * _width)
                + (temp_data["vertical_offset"] ** 2) / (_height * _height)
                <= 1,
                "success",
            ] = True
            only_success = temp_data[(temp_data.success == True)]

            if sel == "Dwell":
                success = False
                if len(only_success) <= 0:
                    print("no touch in ", sub, sel, cur, pos, r, t)
                    trial_summary["error"] = "no touch"
                    summary.loc[len(summary)] = trial_summary
                    continue
                initial_contact_time = only_success.timestamp.values[0]
                walklength = (
                    (
                        temp_data.head_origin_x.diff(1) ** 2
                        + temp_data.head_origin_y.diff(1) ** 2
                        + temp_data.head_origin_z.diff(1) ** 2
                    )
                    .apply(math.sqrt)
                    .sum()
                )
                walk_speed = walklength / (overall_time - initial_contact_time)
                trial_summary["walk_speed"] = walk_speed
                dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                mean_error = dwell_temp.cursor_angular_distance.mean()
                trial_summary["mean_error"] = mean_error
                trial_summary["mean_error_horizontal"] = (
                    dwell_temp.horizontal_offset.mean()
                )
                trial_summary["mean_error_vertical"] = dwell_temp.vertical_offset.mean()
                trial_summary["std_error_horizontal"] = (
                    dwell_temp.horizontal_offset.std()
                )
                trial_summary["std_error_vertical"] = dwell_temp.vertical_offset.std()
                trial_summary["mean_cursor_speed"] = (
                    abs(dwell_temp.cursor_angular_distance.diff())
                    / dwell_temp.timestamp.diff()
                ).mean()
                mrt_df = temp_data[
                    (temp_data.timestamp >= initial_contact_time)
                    & (temp_data.timestamp <= dwell_time + initial_contact_time)
                ]
                mrt = max(list(mrt_df.cursor_angular_distance))
                trial_summary["initial_contact_time"] = initial_contact_time
                trial_summary["required_target_size"] = mrt
                all_success_dwells = []
                for k, g in itertools.groupby(
                    temp_data.iterrows(), key=lambda row: row[1]["success"]
                ):
                    if k == True:
                        df = pd.DataFrame([r[1] for r in g])
                        all_success_dwells.append(df)
                success_dwells = []
                times = []

                target_in_count = len(all_success_dwells)
                trial_summary["target_in_count"] = target_in_count
                target_in_count_per_second = target_in_count / (
                    overall_time - initial_contact_time
                )
                trial_summary["target_in_count_per_second"] = target_in_count_per_second
                for dw in all_success_dwells:
                    time_record = dw.timestamp.values[-1] - dw.timestamp.values[0]
                    times.append(time_record)
                    if time_record >= dwell_time - 1 / 60:
                        success = True
                        success_dwells.append(dw)
                # if success:
                #     trial_summary['first_dwell_time'] = overall_time - 1 + 1.5 / 60
                # if len(success_dwells) >0:
                if success_record[t] == "O":
                    success = True
                trial_summary["success"] = success
                summary.loc[len(summary)] = trial_summary
                continue
            elif sel == "Click":
                trial_summary["final_point_horizontal"] = (
                    temp_data.horizontal_offset.values[-1]
                )
                trial_summary["final_point_vertical"] = (
                    temp_data.vertical_offset.values[-1]
                )
                # temp_data.loc[(temp_data["horizontal_offset"].apply(abs) <= _width) & (
                #         temp_data["vertical_offset"].apply(abs) < _height), "success"] = True

                # if (trial_summary['final_point_horizontal'] <= _width) and (
                #         trial_summary['final_point_vertical'] < _height):
                #     success = True
                # temp_data.loc[
                # (temp_data['horizontal_offset']**2) / (_width*_width) +
                # (temp_data['vertical_offset']**2) / (_height*_height) <=1
                #  ,"success"
                # ]=True
                if (
                    trial_summary["final_point_horizontal"] ** 2 / (_width * _width)
                    + trial_summary["final_point_vertical"] ** 2 / (_height * _height)
                ) <= 1:
                    success = True
                else:
                    success = False
                if success_record[t] == "O":
                    success = True
                trial_summary["success"] = success
                # if len(only_success) <= 0:
                #     # print('no touch in ', sub, sel, cur, pos, r, t)
                #     trial_summary['error'] = "no touch"
                #     summary.loc[len(summary)] = trial_summary
                #     continue
                # initial_contact_time = only_success.timestamp.values[0]
                # trial_summary['initial_contact_time'] = initial_contact_time
                # trial_summary['final_cursor_speed'] = abs(
                #     temp_data.cursor_angular_distance.diff() / temp_data.timestamp.diff())[-12:].mean()
                # walklength = (temp_data.head_origin_x.diff(1) ** 2 + temp_data.head_origin_y.diff(
                #     1) ** 2 + temp_data.head_origin_z.diff(1) ** 2).apply(
                #     math.sqrt).sum()
                # walk_speed = walklength / (overall_time - initial_contact_time)
                # trial_summary['walk_speed'] = walk_speed
                #
                # all_success_dwells = []
                # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['success']):
                #     if k == True:
                #         df = pd.DataFrame([r[1] for r in g])
                #         all_success_dwells.append(df)
                # success_dwells = []
                # times = []
                #
                # target_in_count = len(all_success_dwells)
                # if target_in_count > 0:
                #     trial_summary['target_in_count'] = target_in_count
                #     target_in_count_per_second = target_in_count / (overall_time - initial_contact_time)
                #     trial_summary['target_in_count_per_second'] = target_in_count_per_second
                #     dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                #     trial_summary['mean_cursor_speed'] = (
                #             abs(dwell_temp.cursor_angular_distance.diff()) / dwell_temp.timestamp.diff()).mean()

                summary.loc[len(summary)] = trial_summary
                continue
        except Exception as e:
            print(e, sub, r, cur, sel, pos)
    return summary


@timeit
def dwell_time_analysis(
    dwell_time,
    cursorTypes=None,
    postures=None,
    targets=range(9),
    repetitions=None,
    subjects=range(24),
):
    if postures is None:
        postures = ["STAND", "WALK"]
    if cursorTypes is None:
        cursorTypes = ["HEAD", "EYE", "HAND"]
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]

    fail_count = 0
    summary = pd.DataFrame(
        columns=[
            "dwell_time",
            "subject_num",
            "posture",
            "cursor_type",
            "repetition",
            "target_num",
            "wide",
            "initial_contact_time",
            "target_in_count",
            "target_in_total_time",
            "target_in_mean_time",
            "first_dwell_time",
            "required_target_size",
            "final_speed",
            "min_target_size",
            "best_record",
            "error",
        ]
    )
    for sub_num in subjects:
        for cursor_type in cursorTypes:
            for rep in repetitions:
                for pos in postures:
                    data = read_hololens_data(sub_num, pos, cursor_type, rep)
                    print(dwell_time, sub_num, pos, cursor_type, rep)

                    splited_data = split_target(data)
                    wide = "SMALL" if rep in rep_small else "LARGE"

                    for t in targets:
                        try:
                            trial_summary = {
                                "dwell_time": dwell_time,
                                "subject_num": sub_num,
                                "posture": pos,
                                "cursor_type": cursor_type,
                                "repetition": rep,
                                "target_num": t,
                                "wide": wide,
                            }
                            temp_data = splited_data[t]
                            temp_data.reset_index(inplace=True)
                            temp_data.timestamp -= temp_data.timestamp.values[0]
                            drop_index = temp_data[
                                (temp_data["direction_x"] == 0)
                                & (temp_data["direction_y"] == 0)
                                & (temp_data["direction_z"] == 0)
                            ].index

                            temp_data = check_loss(temp_data, cursor_type)
                            trial_summary["error_frame_count"] = len(
                                temp_data[temp_data["error_frame"] == True]
                            )
                            # # temp_data = temp_data.drop(drop_index)
                            validate, reason = validate_trial_data(
                                temp_data, cursor_type, pos
                            )
                            if not validate:  # in case of invalid trial.
                                trial_summary["error"] = reason
                                print(sub_num, pos, cursor_type, rep, t, reason)
                                summary.loc[len(summary)] = trial_summary
                                continue
                            temp_data["cursor_speed"] = (
                                temp_data.cursor_angular_distance.diff(1)
                                / temp_data.timestamp.diff(1)
                            )
                            temp_data["cursor_speed"] = abs(
                                temp_data.cursor_speed.rolling(
                                    5, min_periods=1, center=True
                                ).mean()
                            )

                            only_success = temp_data[temp_data.success == True]
                            if len(only_success) <= 0:
                                trial_summary["error"] = "no success frame"
                                summary.loc[len(summary)] = trial_summary
                                continue
                                # raise ValueError('no success frames', len(only_success))
                            initial_contact_time = only_success.timestamp.values[0]
                            # mean required target size
                            mrt_df = temp_data[
                                (temp_data.timestamp >= initial_contact_time)
                                & (
                                    temp_data.timestamp
                                    <= dwell_time + initial_contact_time
                                )
                            ]
                            mrt = max(list(mrt_df.angle))
                            trial_summary["required_target_size"] = mrt
                            maxes = []
                            frame = int(dwell_time * 60) - 3
                            # for i in range(len(temp_data.angle) - frame):
                            #     maxes.append(max(temp_data.angle[i:i + frame]))

                            # min_target_size = min(maxes)
                            # trial_summary['min_target_size'] = min_target_size

                            all_success_dwells = []

                            # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_in']):
                            for k, g in itertools.groupby(
                                temp_data.iterrows(), key=lambda row: row[1]["success"]
                            ):
                                # print(k, [t[0] for t in g])
                                # if k == True:
                                if k == True:
                                    # if k == 'Target_' + str(t):
                                    df = pd.DataFrame([r[1] for r in g])
                                    all_success_dwells.append(df)

                            success_dwells = []
                            times = []
                            for dw in all_success_dwells:
                                time_record = (
                                    dw.timestamp.values[-1]
                                    - dw.timestamp.values[0]
                                    + 2.5 / 60
                                )
                                times.append(time_record)
                                if time_record >= dwell_time:
                                    success_dwells.append(dw)
                            best_record = max(times)
                            trial_summary["best_record"] = best_record
                            if len(success_dwells) <= 0:
                                trial_summary["error"] = "no success dwell"
                                summary.loc[len(summary)] = trial_summary
                                continue
                                # raise ValueError('no success dwell', len(success_dwells))
                            time_sum = 0

                            for dw in success_dwells:
                                record = (
                                    dw.timestamp.values[-1]
                                    - dw.timestamp.values[0]
                                    + 2.5 / 60
                                )

                                time_sum += record

                            trial_duration = success_dwells[0].timestamp.values[-1]
                            target_in_count = 0
                            for a in all_success_dwells:
                                if a.timestamp.values[0] < trial_duration:
                                    target_in_count = target_in_count + 1

                            # target_in_count = len(success_dwells)

                            # target_in_total_time = time_sum
                            # target_in_mean_time = time_sum / target_in_count

                            # dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                            """
                            successful trials percentage,
                            time to achieve success
                            number of extra times over 
                            target mean speed during final 100ms
                            mean required-target-size
                            best dwell time
                            """

                            # time to achieve success
                            first_dwell_time = None
                            for index, dw in enumerate(success_dwells):
                                if index == 0:
                                    first_dwell_time = dw.timestamp.values[0]
                            # mean speed during final 100 ms
                            final_speeds = []
                            # frame = int(dwell_time * 60)
                            # final_speed = success_dwells[0].cursor_speed[-frame:].mean()
                            for dw in success_dwells:
                                frame = int(dwell_time * 60)
                                final_speed = dw.cursor_speed[frame - 3 : frame].mean()
                                final_speeds.append(final_speed)
                            mean_final_speed = sum(final_speeds) / len(final_speeds)
                            # mean_final_speed = final_speed
                            trial_summary = {
                                "dwell_time": dwell_time,
                                "subject_num": sub_num,
                                "posture": pos,
                                "cursor_type": cursor_type,
                                "repetition": rep,
                                "target_num": t,
                                "wide": wide,
                                # 'initial_contact_time': initial_contact_time,
                                "target_in_count": target_in_count,
                                # 'target_in_total_time': target_in_total_time,
                                # 'target_in_mean_time': target_in_mean_time,
                                "first_dwell_time": first_dwell_time,
                                "required_target_size": mrt,
                                "final_speed": mean_final_speed,
                                # 'min_target_size': min_target_size,
                                "best_record": best_record,
                                "error": None,
                            }
                            summary.loc[len(summary)] = trial_summary

                        except Exception as e:
                            fail_count += 1
                            error_summary = {
                                "dwell_time": dwell_time,
                                "subject_num": sub_num,
                                "posture": pos,
                                "cursor_type": cursor_type,
                                "repetition": rep,
                                "target_num": t,
                                "wide": wide,
                                # 'required_target_size': mrt,
                                # 'best_record': best_record,
                                "error": e.args,
                            }
                            summary.loc[len(summary)] = error_summary
                            print(
                                dwell_time,
                                sub_num,
                                pos,
                                cursor_type,
                                rep,
                                t,
                                e.args,
                                "fail count",
                                fail_count,
                            )

    final_summary = summary.groupby(
        [summary["posture"], summary["cursor_type"], summary["wide"]]
    ).mean()
    final_summary.to_csv("dwell_time_summary" + str(dwell_time) + ".csv")
    summary.to_csv("dwell_time_Rawsummary" + str(dwell_time) + ".csv")
    return summary


def watch_errors():
    subjects = range(24)
    for sub_num in subjects:
        rep_small = [0, 2, 4, 6, 8]
        rep_large = [1, 3, 5, 7, 9]
        draw_plot = False
        cursorTypes = ["HEAD", "EYE", "HAND"]

        postures = ["WALK", "STAND"]
        targets = range(9)
        repetitions = [4, 5, 6, 7, 8, 9]
        # repetitions = range(10)
        fail_count = 0
        summary = pd.DataFrame(
            columns=[
                "subject_num",
                "posture",
                "cursor_type",
                "repetition",
                "target_num",
                "wide",
                "mean_offset",
                "std_offset",
                "overall_mean_offset",
                "overall_std_offset",
                "initial_contact_time",
                "target_in_count",
                "target_in_total_time",
                "target_in_mean_time",
                "mean_offset_horizontal",
                "mean_offset_vertical",
                "std_offset_horizontal",
                "std_offset_vertical",
                "error",
            ]
        )
        outlier_count = 0
        loss_count = 0
        total_count = 0
        for cursor_type in cursorTypes:
            for rep in repetitions:
                for pos in postures:
                    data = read_hololens_data(sub_num, pos, cursor_type, rep)

                    wide = "SMALL" if rep in rep_small else "LARGE"
                    splited_data = split_target(data)
                    # data = data[5:]
                    for t in targets:
                        try:
                            total_count += 1
                            validation, reason = validate_trial_data(
                                splited_data[t], cursor_type, pos
                            )
                            if validation == False:
                                if reason == "loss":
                                    loss_count += 1
                                elif reason == "jump":
                                    outlier_count += 1
                            # temp_data = splited_data[t]
                            # temp_data.reset_index(inplace=True)
                            # temp_data.timestamp -= temp_data.timestamp.values[0]
                            # drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
                            #         temp_data['direction_z'] == 0)].index
                            # outlier = temp_data[
                            #               (temp_data['target_horizontal_velocity'] > 10) | (
                            #                       temp_data['target_horizontal_velocity'] < -10)][5:]
                            # if len(outlier.timestamp.values) > 1:
                            #     # print(sub_num, pos, cursor_type, rep,t,len(outlier.timestamp.values),outlier.timestamp.values)
                            #     outlier_count += 1
                        except Exception as e:
                            print("error", sub_num, pos, cursor_type, rep, e.args)
        print(
            "outlier movement per person",
            sub_num,
            "jump",
            outlier_count,
            "loss",
            loss_count,
            total_count,
        )


def plt_confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse(x, y, n_std=1.96, size=100):
    """
    Get the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    size : int
        Number of points defining the ellipse
    Returns
    -------
    String containing an SVG path for the ellipse

    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)]
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = (
        ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix
    )

    path = f"M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}"
    for k in range(1, len(ellipse_coords)):
        path += f"L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}"
    path += " Z"
    return path


def find_outliers():
    summary_dataframe = visualize_offsets(show_plot=False)
    head_data = summary_dataframe[
        (summary_dataframe["posture"] == "WALK")
        & (summary_dataframe["cursor_type"] == "HEAD")
    ]
    head_horizontals = []
    for hh in head_data.horizontal.values:
        head_horizontals += list(hh)
    parameters = {}
    for ct, pos, ax in itertools.product(
        ["EYE", "HAND", "HEAD"], ["WALK", "STAND"], ["horizontal", "vertical"]
    ):
        head_data = summary_dataframe[
            (summary_dataframe["posture"] == pos)
            & (summary_dataframe["cursor_type"] == ct)
        ]
        head_horizontals = []
        for hh in head_data[ax].values:
            head_horizontals += list(hh)
        import seaborn as sns

        sns.displot(head_horizontals, kde=True)
        head_horizontals = np.array(head_horizontals)
        sigma = head_horizontals.std()
        mean = head_horizontals.mean()
        # plt.axvline(mean)
        # plt.axvline(mean+2*sigma)
        # plt.axvline(mean-2*sigma)
        # plt.axvline(mean+3*sigma)
        # plt.axvline(mean-3*sigma)
        # plt.title(str(ct)+'_'+str(pos)+'_'+str(ax))

        # plt.show()
        print(ct, pos, ax, mean, sigma, 3 * sigma)
        parameters[(ct, pos, ax)] = sigma
        """
        EYE WALK horizontal 0.24722741899600512 4.420237751534142 13.260713254602425
        EYE WALK vertical -0.07572955993240958 2.4375580926867078 7.312674278060124
        EYE STAND horizontal 0.20836864617779696 1.5635038623192548 4.690511586957764
        EYE STAND vertical 0.18935830234235748 1.491778058469321 4.475334175407963
        HAND WALK horizontal -0.09866966003016135 6.521336309396893 19.564008928190677
        HAND WALK vertical -0.26136047800482975 1.6178699940290733 4.85360998208722
        HAND STAND horizontal 0.04440054874312414 1.2868251691549768 3.8604755074649306
        HAND STAND vertical 0.05941293945469808 1.3437840646867873 4.0313521940603625
        HEAD WALK horizontal -0.0040063012538855474 5.0511439371221885 15.153431811366566
        HEAD WALK vertical 0.3109466494847721 2.3182985184738376 6.954895555421513
        HEAD STAND horizontal -0.0398596984122698 1.303755389483091 3.9112661684492736
        HEAD STAND vertical 0.1434393747167268 1.5906082672928836 4.771824801878651
        """


def easing(data, factor):
    eased = []
    for idx, d in enumerate(list(data)):
        if idx == 0:
            eased.append(d)
        else:
            eased.append(eased[-1] * (1 - factor) + d * factor)
    return eased


def movingAverage(data, window):
    data = data.rolling(min_periods=1, window=window)
    return data


def weightedAverage(data, window):
    output = []
    data = list(data)
    potential = 0
    for i in range(len(data)):
        if i == 0:
            output.append(data[i])
            continue
        elif i < window:
            length = i
        elif i > len(data) - window:
            length = len(data) - i
        else:
            length = window
        temp = data[i - length : i]
        result = 0
        for j in range(len(temp)):
            result += (j + 1) * np.array(temp[j])
        result = result / sum(range(len(temp) + 1))
        output.append(result)
    return pd.Series(output)


def smoothTriangle(data, degree, dropVals=False):
    triangle = np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i : i + len(triangle)] * triangle
        smoothed.append(sum(point) / sum(triangle))
    if dropVals:
        return smoothed
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def TriangleDataframe(data, window):
    # data.cursor_vertical_angle = weightedAverage(data.cursor_vertical_angle, window)
    # data.cursor_horizontal_angle = weightedAverage(data.cursor_horizontal_angle, window)
    # data.cursor_rotation = weightedAverage(data.cursor_rotation,window)
    data.direction_x = weightedAverage(data.direction_x, window)
    data.direction_y = weightedAverage(data.direction_y, window)
    data.direction_z = weightedAverage(data.direction_z, window)
    data["cursor_rotation"] = data.apply(
        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1
    )
    data["cursor_horizontal_angle"] = data.apply(lambda x: x.cursor_rotation[1], axis=1)
    data["cursor_vertical_angle"] = data.apply(lambda x: x.cursor_rotation[0], axis=1)
    data["horizontal_offset"] = (
        data.target_horizontal_angle - data.cursor_horizontal_angle
    ).apply(correct_angle)
    data["vertical_offset"] = (
        data.target_vertical_angle - data.cursor_vertical_angle
    ).apply(correct_angle)
    data["angle"] = (data.horizontal_offset**2 + data.vertical_offset**2).apply(
        math.sqrt
    )
    data["success"] = data.angle < data.max_angle
    data["abs_horizontal_offset"] = data["horizontal_offset"].apply(abs)
    data["abs_vertical_offset"] = data["vertical_offset"].apply(abs)
    return data


def MovingAverage(data, window):
    # data.cursor_vertical_angle = weightedAverage(data.cursor_vertical_angle, window)
    # data.cursor_horizontal_angle = weightedAverage(data.cursor_horizontal_angle, window)
    # data.cursor_rotation = weightedAverage(data.cursor_rotation,window)
    # data.direction_x = movingAverage(data.direction_x, window)
    # data.direction_y = movingAverage(data.direction_y, window)
    # data.direction_z = movingAverage(data.direction_z, window)
    data.direction_x = data.direction_x.rolling(min_periods=1, window=window).mean()
    data.direction_y = data.direction_y.rolling(min_periods=1, window=window).mean()
    data.direction_z = data.direction_z.rolling(min_periods=1, window=window).mean()
    data["cursor_rotation"] = data.apply(
        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1
    )
    data["cursor_horizontal_angle"] = data.apply(lambda x: x.cursor_rotation[1], axis=1)
    data["cursor_vertical_angle"] = data.apply(lambda x: x.cursor_rotation[0], axis=1)
    data["horizontal_offset"] = (
        data.target_horizontal_angle - data.cursor_horizontal_angle
    ).apply(correct_angle)
    data["vertical_offset"] = (
        data.target_vertical_angle - data.cursor_vertical_angle
    ).apply(correct_angle)
    data["angle"] = (data.horizontal_offset**2 + data.vertical_offset**2).apply(
        math.sqrt
    )
    data["success"] = data.angle < data.max_angle
    data["abs_horizontal_offset"] = data["horizontal_offset"].apply(abs)
    data["abs_vertical_offset"] = data["vertical_offset"].apply(abs)
    return data
