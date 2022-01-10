import math

import matplotlib.patches
import pandas as pd

from FileHandling import *

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def collect_offsets(sub_num, cursorTypes=None, postures=None, targets=range(9),
                    repetitions=None):
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    if postures is None:
        postures = ['STAND', 'WALK']
    if cursorTypes is None:
        cursorTypes = ['HEAD', 'EYE', 'HAND']
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]
    summary = pd.DataFrame(
        columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide', 'horizontal_offset_list',
                 'vertical_offset_list', 'error'])
    for cursor_type, rep, pos in itertools.product(cursorTypes, repetitions, postures):
        data = read_hololens_data(sub_num, pos, cursor_type, rep)
        splited_data = split_target(data)
        wide = 'SMALL' if rep in rep_small else 'LARGE'
        for t in targets:
            try:
                trial_summary = {'subject_num': sub_num,
                                 'posture': pos,
                                 'cursor_type': cursor_type,
                                 'repetition': rep,
                                 'target_num': t,
                                 'wide': wide,
                                 }
                temp_data = splited_data[t]
                temp_data.reset_index(inplace=True)
                temp_data.timestamp -= temp_data.timestamp.values[0]
                validate, reason = validate_trial_data(temp_data)
                if not validate:  # in case of invalid trial.
                    trial_summary['error'] = reason
                    summary.loc[len(summary)] = trial_summary
                    continue
                drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
                        temp_data['direction_z'] == 0)].index
                # if len(drop_index) > 0:
                #     print('drop length', len(drop_index), sub_num, pos, cursor_type, rep, t)
                #     # raise ValueError
                temp_data = temp_data.drop(drop_index)
                only_success = temp_data[temp_data.target_name == "Target_" + str(t)]
                if len(only_success) <= 0:
                    raise ValueError('no success frames', len(only_success))
                initial_contact_time = only_success.timestamp.values[0]
                dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                trial_summary['horizontal_offset_list'] = list(dwell_temp.horizontal_offset)
                trial_summary['vertical_offset_list'] = list(dwell_temp.vertical_offset)
                summary.loc[len(summary)] = trial_summary
            except Exception as e:
                error_summary = {'subject_num': sub_num,
                                 'posture': pos,
                                 'cursor_type': cursor_type,
                                 'repetition': rep,
                                 'target_num': t,
                                 'wide': wide,
                                 'error': e.args
                                 }
                summary.loc[len(summary)] = error_summary
    # final_summary = summary.groupby([summary['posture'], summary['cursor_type'], summary['wide']]).mean()
    # final_summary.to_csv('summary' + str(sub_num) + '.csv')
    summary.to_pickle('offset_lists' + str(sub_num) + '.pkl')
    return summary


def visualize_offsets(show_plot=True):
    subjects = range(24)
    dfs = []
    for subject in subjects:
        summary_subject = pd.read_pickle('offset_lists' + str(subject) + '.pkl')
        dfs.append(summary_subject)
    summary = pd.concat(dfs)
    cursorTypes = ['HEAD', 'EYE', 'HAND']
    postures = ['STAND', 'WALK']
    # print('total failure',summary.isnull().mean_offset.sum(),'/',len(summary.index))

    errors = summary[summary.error.isna() == False]
    print(errors.groupby(errors['error']).subject_num.count())
    # if show_plot:
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, dpi=100)
    # plot over each posture,cursor types
    summary_dict = {}
    summary_dataframe = pd.DataFrame(columns=[
        'posture', 'cursor_type', 'wide', 'horizontal', 'vertical', 'target_num'
    ])
    for posture, cursor_type, target_num, wide in itertools.product(postures, cursorTypes, range(9),
                                                                    ['LARGE', 'SMALL']):
        temp_summary = summary[
            (summary.posture == posture) & (summary.cursor_type == cursor_type) & (summary.target_num == target_num) & (
                    summary.wide == wide)]
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
            'posture': posture,
            'cursor_type': cursor_type,
            'wide': wide,
            'horizontal': hs,
            'vertical': vs,
            'target_num': target_num
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

        colorset = ['maroon', 'orangered', 'darkorange', 'olive', 'yellowgreen', 'darkolivegreen', 'turquoise',
                    'deepskyblue',
                    'dodgerblue']
        postures = ['STAND', 'WALK']
        # postures = ['WALK']
        sigma_multiple = 3
        for posture, cursor_type in itertools.product(postures, ['EYE', 'HAND', 'HEAD']):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            size_fig, size_ax = plt.subplots(1, 1, figsize=(10, 10))
            for t in range(9):
                for w in ['SMALL', 'LARGE']:
                    if w == 'LARGE':
                        wide = 14.04
                    else:
                        wide = 7.125
                    # wide = 10
                    x_offset = wide * math.sin(t * math.pi / 9 * 2)
                    y_offset = wide * math.cos(t * math.pi / 9 * 2)
                    h_raw = summary_dataframe.loc[
                        (summary_dataframe['posture'] == posture) & (summary_dataframe['wide'] == w) & (
                                summary_dataframe['cursor_type'] == cursor_type) & (
                                summary_dataframe['target_num'] == t)]['horizontal'].values[0]
                    v_raw = summary_dataframe.loc[
                        (summary_dataframe['posture'] == posture) & (summary_dataframe['wide'] == w) & (
                                summary_dataframe['cursor_type'] == cursor_type) & (
                                summary_dataframe['target_num'] == t)]['vertical'].values[0]
                    h = []
                    v = []
                    for i in range(len(h_raw)):
                        if not (-sigma_multiple * sigmas[(cursor_type, posture, 'horizontal')] < h_raw[
                            i] < sigma_multiple *
                                sigmas[
                                    (cursor_type, posture, 'horizontal')]):
                            continue
                        elif not (-sigma_multiple * sigmas[(cursor_type, posture, 'vertical')] < v_raw[
                            i] < sigma_multiple *
                                  sigmas[
                                      (cursor_type, posture, 'vertical')]):
                            continue
                        h.append(h_raw[i])
                        v.append(v_raw[i])
                    h = np.array(h)
                    v = np.array(v)
                    # sns.kdeplot(x=h + x_offset, y=v + y_offset, fill=True, ax=ax)
                    ax.scatter(x_offset, y_offset, s=100, c=colorset[t], marker='x')
                    # ax.scatter(h + x_offset, v + y_offset, s=0.5, alpha=0.05, c=colorset[t])
                    plt_confidence_ellipse(h + x_offset, v + y_offset, ax, 2, edgecolor=colorset[t], linestyle='--',
                                           linewidth=3)
                    size_ax.scatter(x_offset, y_offset, s=100, c=colorset[t], marker='x')
                    import matplotlib.patches as patches
                    width = h.mean() + 2 * h.std()
                    height = v.mean() + 2 * v.std()
                    size_ax.add_patch(
                        patches.Rectangle(
                            (x_offset - width, y_offset - height)
                            , 2 * width, 2 * height, edgecolor=colorset[t], fill=False
                        )
                    )
            plt.title(str(posture) + "," + str(cursor_type) + '-' + str(sigma_multiple))
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.show()
    return summary_dataframe


def summarize_subject(sub_num, cursorTypes=None, postures=None, targets=range(9),
                      repetitions=None, pilot=False, savefile=True):
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    if postures is None:
        postures = ['STAND', 'WALK']
    if cursorTypes is None:
        cursorTypes = ['HEAD', 'EYE', 'HAND']
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]
    draw_plot = False

    # repetitions = range(10)
    fail_count = 0
    summary = pd.DataFrame(
        columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide', 'mean_offset',
                 'std_offset', 'overall_mean_offset', 'overall_std_offset',
                 'initial_contact_time', 'target_in_count', 'target_in_total_time', 'target_in_mean_time',
                 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal', 'std_offset_vertical',
                 'mean_abs_offset_horizontal', 'mean_abs_offset_vertical', 'std_abs_offset_horizontal',
                 'std_abs_offset_vertical', 'longest_dwell_time',
                 'movement_length',
                 'entering_position',
                 'error'])
    for cursor_type in cursorTypes:
        for rep in repetitions:
            for pos in postures:
                data = read_hololens_data(sub_num, pos, cursor_type, rep, False, pilot)
                splited_data = split_target(data)
                wide = 'SMALL' if rep in rep_small else 'LARGE'

                for t in targets:
                    try:
                        trial_summary = {'subject_num': sub_num,
                                         'posture': pos,
                                         'cursor_type': cursor_type,
                                         'repetition': rep,
                                         'target_num': t,
                                         'wide': wide,
                                         }
                        temp_data = splited_data[t]
                        temp_data.reset_index(inplace=True)
                        temp_data.timestamp -= temp_data.timestamp.values[0]

                        drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
                                temp_data['direction_z'] == 0)].index
                        validate, reason = validate_trial_data(temp_data)
                        if not validate:  # in case of invalid trial.
                            trial_summary['error'] = reason
                            summary.loc[len(summary)] = trial_summary
                            continue
                        if len(drop_index) > 0:
                            loss_indices = set(list(drop_index) + list(drop_index + 1) + list(drop_index + 2))
                            if len(temp_data) in loss_indices:
                                loss_indices.remove(len(temp_data))
                            if len(temp_data) + 1 in loss_indices:
                                loss_indices.remove(len(temp_data) + 1)
                            temp_data.loc[loss_indices] = np.nan
                            temp_data = temp_data.interpolate()


                        # temp_data = temp_data.drop(drop_index)
                        temp_data['cursor_speed'] = temp_data.cursor_angular_distance.diff(
                            1) / temp_data.timestamp.diff(1)
                        temp_data['cursor_speed'] = abs(
                            temp_data.cursor_speed.rolling(10, min_periods=1, center=True).mean())


                        only_success = temp_data[temp_data.cursor_angular_distance < default_target_size]
                        if len(only_success) <= 0:
                            raise ValueError('no success frames', len(only_success))
                        initial_contact_time = only_success.timestamp.values[0]

                        success_dwells = []
                        temp_data['target_in'] = temp_data['cursor_angular_distance'] < default_target_size
                        for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_in']):
                        # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_name']):
                            # print(k, [t[0] for t in g])
                            if k==True:
                            # if k == 'Target_' + str(t):
                                df = pd.DataFrame([r[1] for r in g])
                                success_dwells.append(df)
                        time_sum = 0
                        times = []
                        for dw in success_dwells:
                            current_dwell_time = dw.timestamp.values[-1] - dw.timestamp.values[0]
                            time_sum += current_dwell_time
                            times.append(current_dwell_time)

                        target_in_count = len(success_dwells)
                        target_in_total_time = time_sum
                        target_in_mean_time = time_sum / target_in_count
                        # TODO
                        longest_dwell_time = max(times)
                        targeting = temp_data[temp_data.timestamp <= initial_contact_time]
                        movement = (targeting.horizontal_offset.diff(1) ** 2 + targeting.vertical_offset.diff(
                            1) ** 2).apply(math.sqrt)
                        movement_length = movement.sum()
                        contact_frame = temp_data[temp_data.timestamp == initial_contact_time]
                        x = contact_frame.horizontal_offset.values[0]
                        y = contact_frame.vertical_offset.values[0]
                        entering_position = (-x, y)

                        dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                        #
                        mean_offset_horizontal = dwell_temp.horizontal_offset.mean()
                        std_offset_horizontal = dwell_temp.horizontal_offset.std()
                        mean_offset_vertical = dwell_temp.vertical_offset.mean()
                        std_offset_vertical = dwell_temp.vertical_offset.std()

                        trial_summary = {'subject_num': sub_num,
                                         'posture': pos,
                                         'cursor_type': cursor_type,
                                         'repetition': rep,
                                         'target_num': t,
                                         'wide': wide,
                                         'mean_offset': dwell_temp.cursor_angular_distance.mean(),
                                         'std_offset': dwell_temp.cursor_angular_distance.std(),
                                         'overall_mean_offset': temp_data.cursor_angular_distance.mean(),
                                         'overall_std_offset': temp_data.cursor_angular_distance.std(),
                                         'initial_contact_time': initial_contact_time,
                                         'target_in_count': float(target_in_count),
                                         'target_in_total_time': target_in_total_time,
                                         'target_in_mean_time': target_in_mean_time,
                                         'mean_offset_horizontal': mean_offset_horizontal,
                                         'mean_offset_vertical': mean_offset_vertical,
                                         'std_offset_horizontal': std_offset_horizontal,
                                         'std_offset_vertical': std_offset_vertical,
                                         'mean_abs_offset_horizontal': dwell_temp.abs_horizontal_offset.mean(),
                                         'mean_abs_offset_vertical': dwell_temp.abs_vertical_offset.mean(),
                                         'std_abs_offset_horizontal': dwell_temp.abs_horizontal_offset.std(),
                                         'std_abs_offset_vertical': dwell_temp.abs_vertical_offset.std(),
                                         'longest_dwell_time': longest_dwell_time,
                                         'movement_length': movement_length,
                                         'entering_position': entering_position,
                                         'error': None
                                         }
                        summary.loc[len(summary)] = trial_summary
                        # smalls.loc[len(smalls)] = [sub_num, 'STAND', cursor_type, rep, t, temp_data.cursor_angular_distance.mean(),
                        #                            temp_data.cursor_angular_distance.std()]
                    except Exception as e:
                        fail_count += 1
                        error_summary = {'subject_num': sub_num,
                                         'posture': pos,
                                         'cursor_type': cursor_type,
                                         'repetition': rep,
                                         'target_num': t,
                                         'wide': wide,
                                         'error': e.args
                                         }
                        summary.loc[len(summary)] = error_summary
                        print(sub_num, pos, cursor_type, rep, t, e.args, 'fail count', fail_count)

    final_summary = summary.groupby([summary['posture'], summary['cursor_type'], summary['wide']]).mean()
    if savefile:
        if pilot:
            final_summary.to_csv('nocursor_summary' + str(sub_num) + '.csv')
            summary.to_csv('nocursor_Rawsummary' + str(sub_num) + '.csv')
        else:
            final_summary.to_csv('summary' + str(sub_num) + '.csv')
            summary.to_csv('Rawsummary' + str(sub_num) + '.csv')
    return summary


def visualize_summary(show_plot=True,show_distribution=False):
    subjects = range(24)
    dfs = []
    for subject in subjects:
        summary_subject = pd.read_csv('Rawsummary' + str(subject) + '.csv')
        dfs.append(summary_subject)
    summary = pd.concat(dfs)
    errors = summary[summary.error.isna() == False]
    print(errors.groupby(errors['error']).subject_num.count())

    def cart2pol(x, y):
        z = x + y * 1j
        r, theta = np.abs(z), np.angle(z)
        return (r, theta)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    from ast import literal_eval
    for pos in ['STAND']:
        for t in range(9):
            entering = summary[(summary.target_num == t) & (summary.posture==pos)].entering_position
            a = entering[entering.notna()]
            entering_positions = a.apply(literal_eval).values
            x, y = zip(*entering_positions)
            spherical = list(map(cart2pol, x, y))
            sx,sy = zip(*spherical)
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')

            colors = sy
            c = ax.scatter(sy, sx,marker='.', alpha=0.75)

            default_r,default_theta = cart2pol(directions[t][0],directions[t][1])
            ax.scatter(default_theta + math.pi, 1.5, marker='x')


            plt.title(pos + " - "+str(t))
            plt.show()


    if (show_plot == True):
        fs = summary.groupby([summary.posture, summary.cursor_type, summary.wide]).mean()

        fs.to_csv('total_basic_summary.csv')
        fs = pd.read_csv('total_basic_summary.csv')
        parameters = list(fs.columns)
        remove_columns = ['Unnamed: 0', 'subject_num', 'repetition', 'target_num']
        for removal in remove_columns:
            parameters.remove(removal)

        fig = px.bar(fs, x='cursor_type', y=parameters, barmode='group', facet_row='wide', facet_col='posture',
                     title='total basic summary')
        fig.show()
        fs_overall = summary.groupby([summary.posture, summary.cursor_type]).mean()
        # estimated target size
        fs_overall['estimated_width'] = fs_overall.mean_offset_horizontal.apply(
            abs) + 2 * fs_overall.std_offset_horizontal
        fs_overall['estimated_height'] = fs_overall.mean_offset_vertical.apply(abs) + 2 * fs_overall.std_offset_vertical
        fs_overall.to_csv('total_basic_summary_overall.csv')
        fs_overall = pd.read_csv('total_basic_summary_overall.csv')
        parameters.remove('wide')
        fig = px.bar(fs_overall, x='cursor_type', y=parameters + ['estimated_width', 'estimated_height'],
                     barmode='group',
                     facet_col='posture',
                     title='total basic summary')
        fig.show()
        # wide = 20
        if show_distribution:
            for posture in ['WALK', 'STAND']:
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
                    for idx, cursor_type in enumerate(['EYE', 'HAND', 'HEAD']):
                        for w in ['LARGE', 'SMALL']:
                            if w == 'LARGE':
                                wide = 14.04
                            else:
                                wide = 7.125
                            cursor_data = summary[
                                (summary['posture'] == posture) & (summary['cursor_type'] == cursor_type) & (
                                        summary['target_num'] == t) & (summary['wide'] == w)]
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
                                    mode='markers',
                                    marker={'color': color}
                                    , opacity=0.1
                                )
                            )
                            fig.add_shape(type='path',
                                          path=confidence_ellipse(xs,
                                                                  ys),
                                          line={'dash': 'dot'},
                                          line_color=color)
                            colorset = ['maroon', 'orangered', 'darkorange', 'olive', 'yellowgreen', 'darkolivegreen',
                                        'turquoise', 'deepskyblue', 'dodgerblue']
                            estimated_half_width = abs(
                                cursor_data.mean_offset_horizontal.mean()) + 2 * cursor_data.std_offset_horizontal.mean()
                            estimated_half_height = abs(
                                cursor_data.mean_offset_vertical.mean()) + 2 * cursor_data.std_offset_vertical.mean()
                            import matplotlib.patches as patches
                            plt_ax.add_patch(
                                patches.Rectangle(
                                    (x_offset - estimated_half_width, y_offset - estimated_half_height),
                                    2 * estimated_half_width,
                                    2 * estimated_half_height,
                                    edgecolor=colorset[t],
                                    fill=False
                                )
                            )

                fig.update_yaxes(scaleanchor='x', scaleratio=1)
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
    cursorTypes = ['HEAD', 'EYE', 'HAND']

    postures = ['WALK']
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
                wide = 'SMALL' if rep in rep_small else 'LARGE'
                # for t in targets:
                try:

                    data['cursor_rotation'] = data.apply(
                        lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                    data['target_rotation'] = data.apply(
                        lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                              x.target_position_z - x.origin_z), axis=1)
                    data['cursor_horizontal'] = data.apply(
                        lambda x: math.sin(math.radians(x.cursor_rotation[1])), axis=1
                    )
                    data['cursor_vertical'] = data.apply(
                        lambda x: x.cursor_rotation[0], axis=1
                    )
                    data['target_horizontal'] = data.apply(
                        lambda x: math.sin(math.radians(x.target_rotation[1])), axis=1
                    )
                    data['target_vertical'] = data.apply(
                        lambda x: x.target_rotation[0], axis=1
                    )
                    data['horizontal'] = data.apply(
                        lambda x: change_angle(x.cursor_horizontal - x.target_horizontal), axis=1
                    )
                    data['vertical'] = data.apply(
                        lambda x: change_angle(x.cursor_vertical - x.target_vertical), axis=1
                    )
                    data['target_vertical_velocity'] = (
                            data['target_vertical'].diff(1) / data['timestamp'].diff(1)).rolling(30).mean()
                    data['target_horizontal_velocity'] = (data['target_horizontal'].diff(1) / data['timestamp'].diff(1))
                    data['target_horizontal_acc'] = (
                            data['target_horizontal_velocity'].diff(1) / data['timestamp'].diff(1))
                    data['walking_speed'] = (
                            (data['head_position_x'].diff(1).pow(2) + data['head_position_z'].diff(1).pow(2)).apply(
                                math.sqrt) / data[
                                'timestamp'].diff(1)).rolling(6, min_periods=1).mean()
                    data['walking_acc'] = (data['walking_speed'].diff(1) / data['timestamp'].diff(1))
                    data['target_speed'] = (
                            (data['target_position_x'].diff(1).pow(2) + data['target_position_z'].diff(1).pow(2)).apply(
                                math.sqrt) / data[
                                'timestamp'].diff(1)).rolling(6, min_periods=1).mean()
                    # target_horizontal_acc += list(data['target_horizontal_acc'])
                    # walking_acc += list(data['walking_acc'])
                    # walking_speeds += list(data['walking_speed'])
                    data['trial_check'] = data['end_num'].diff(1)
                    data_without_change = data[(data.trial_check == 0)][5:]
                    # plt.plot(data_without_change.timestamp, data_without_change.target_horizontal_velocity)
                    target_horizotal_vels += list(data_without_change.target_horizontal_velocity)
                    fail_check = data_without_change[(data_without_change.target_horizontal_velocity > 5) | (
                            data_without_change.target_horizontal_velocity < -5)]
                    if len(fail_check) > 0:
                        fail_count += len(data.groupby(data.end_num).count())
                except Exception as e:
                    print(sub_num, pos, cursor_type, rep, e.args, 'fail count', fail_count)
    # return target_horizontal_acc,walking_acc,walking_speeds
    return target_horizotal_vels, fail_count


def target_size_analysis(target_size, cursorTypes=None, postures=None, targets=range(9),
                         repetitions=None, subjects=range(24)):
    if postures is None:
        postures = ['STAND', 'WALK']
    if cursorTypes is None:
        cursorTypes = ['HEAD', 'EYE', 'HAND']
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
        columns=['target_size', 'subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide',
                 'mean_offset',
                 'std_offset', 'overall_mean_offset', 'overall_std_offset',
                 'initial_contact_time', 'target_in_count', 'target_in_total_time', 'target_in_mean_time',
                 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal', 'std_offset_vertical',
                 'longes t_dwell',
                 'error'])
    for sub_num in subjects:
        for cursor_type in cursorTypes:
            for rep in repetitions:
                for pos in postures:
                    data = read_hololens_data(sub_num, pos, cursor_type, rep)
                    splited_data = split_target(data)
                    wide = 'SMALL' if rep in rep_small else 'LARGE'

                    for t in targets:
                        try:
                            trial_summary = {
                                'target_size': target_size,
                                'subject_num': sub_num,
                                'posture': pos,
                                'cursor_type': cursor_type,
                                'repetition': rep,
                                'target_num': t,
                                'wide': wide,
                            }
                            temp_data = splited_data[t]
                            temp_data.reset_index(inplace=True)
                            temp_data.timestamp -= temp_data.timestamp.values[0]
                            validate, reason = validate_trial_data(temp_data)
                            if not validate:  # in case of invalid trial.
                                trial_summary['error'] = reason
                                summary.loc[len(summary)] = trial_summary
                                continue
                            drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
                                    temp_data['direction_z'] == 0)].index
                            # if len(drop_index) > 0:
                            #     print('drop length', len(drop_index), sub_num, pos, cursor_type, rep, t)
                            #     # raise ValueError
                            temp_data = temp_data.drop(drop_index)
                            # only_success = temp_data[temp_data.target_name == "Target_" + str(t)]
                            only_success = temp_data[temp_data.cursor_angular_distance < target_size]

                            if len(only_success) <= 0:
                                raise ValueError('no success frames', len(only_success))
                            initial_contact_time = only_success.timestamp.values[0]

                            success_dwells = []
                            temp_data['target_in'] = temp_data['cursor_angular_distance'] < target_size
                            for k, g in itertools.groupby(temp_data.iterrows(),
                                                          key=lambda row: row[1]['target_in']):
                                # print(k, [t[0] for t in g])
                                if k == True:
                                    df = pd.DataFrame([r[1] for r in g])
                                    success_dwells.append(df)
                            time_sum = 0
                            times = []
                            for dw in success_dwells:
                                time_sum += dw.timestamp.values[-1] - dw.timestamp.values[0]
                                times.append(dw.timestamp.values[-1] - dw.timestamp.values[0])
                            longest_dwell = max(times)
                            target_in_count = len(success_dwells)
                            target_in_total_time = time_sum
                            target_in_mean_time = time_sum / target_in_count

                            dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
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

                            trial_summary = {'target_size': target_size, 'subject_num': sub_num,
                                             'posture': pos,
                                             'cursor_type': cursor_type,
                                             'repetition': rep,
                                             'target_num': t,
                                             'wide': wide,
                                             'mean_offset': dwell_temp.cursor_angular_distance.mean(),
                                             'std_offset': dwell_temp.cursor_angular_distance.std(),
                                             'overall_mean_offset': temp_data.cursor_angular_distance.mean(),
                                             'overall_std_offset': temp_data.cursor_angular_distance.mean(),
                                             'initial_contact_time': initial_contact_time,
                                             'target_in_count': float(target_in_count),
                                             'target_in_total_time': target_in_total_time,
                                             'target_in_mean_time': target_in_mean_time,
                                             'mean_offset_horizontal': mean_offset_horizontal,
                                             'mean_offset_vertical': mean_offset_vertical,
                                             'std_offset_horizontal': std_offset_horizontal,
                                             'std_offset_vertical': std_offset_vertical,
                                             'longest_dwell': longest_dwell,
                                             'error': None
                                             }
                            summary.loc[len(summary)] = trial_summary
                            # smalls.loc[len(smalls)] = [sub_num, 'STAND', cursor_type, rep, t, temp_data.cursor_angular_distance.mean(),
                            #                            temp_data.cursor_angular_distance.std()]
                        except Exception as e:
                            fail_count += 1
                            error_summary = {
                                'target_size': target_size, 'subject_num': sub_num,
                                'posture': pos,
                                'cursor_type': cursor_type,
                                'repetition': rep,
                                'target_num': t,
                                'wide': wide,
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
                                'error': e.args
                            }
                            summary.loc[len(summary)] = error_summary
                            print(sub_num, pos, cursor_type, rep, t, e.args, 'fail count', fail_count)
                            # plt.plot(temp_data.timestamp, temp_data.cursor_angular_distance, label=str(pos))
                            # # plt.plot(walk_temp.timestamp, walk_temp.cursor_angular_distance, label='walk')
                            # plt.axvline(initial_contact_time)
                            # # plt.axvline(walk_initial_contact_time)
                            # # plt.text(initial_contact_time,0,'initial contact time',)
                            # plt.legend()
                            # plt.title(f'{cursor_type} angular offset from :target' + str(t))
                            # plt.show()
    final_summary = summary.groupby([summary['posture'], summary['cursor_type'], summary['wide']]).mean()
    final_summary.to_csv('target_size_summary' + str(target_size) + '.csv')
    summary.to_csv('target_size_Rawsummary' + str(target_size) + '.csv')
    return summary


def dwell_time_analysis(dwell_time, cursorTypes=None, postures=None, targets=range(9),
                        repetitions=None, subjects=range(24)):
    if postures is None:
        postures = ['STAND', 'WALK']
    if cursorTypes is None:
        cursorTypes = ['HEAD', 'EYE', 'HAND']
    if repetitions is None:
        repetitions = [4, 5, 6, 7, 8, 9]
    rep_small = [0, 2, 4, 6, 8]
    rep_large = [1, 3, 5, 7, 9]

    fail_count = 0
    summary = pd.DataFrame(
        columns=['dwell_time', 'subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide',
                 'initial_contact_time', 'target_in_count', 'target_in_total_time', 'target_in_mean_time',
                 'first_dwell_time', 'required_target_size', 'mean_final_speed', 'error'])
    for sub_num in subjects:
        for cursor_type in cursorTypes:
            for rep in repetitions:
                for pos in postures:
                    data = read_hololens_data(sub_num, pos, cursor_type, rep)

                    splited_data = split_target(data)
                    wide = 'SMALL' if rep in rep_small else 'LARGE'

                    for t in targets:
                        try:
                            trial_summary = {
                                'dwell_time': dwell_time,
                                'subject_num': sub_num,
                                'posture': pos,
                                'cursor_type': cursor_type,
                                'repetition': rep,
                                'target_num': t,
                                'wide': wide,
                            }
                            temp_data = splited_data[t]
                            temp_data.reset_index(inplace=True)
                            temp_data.timestamp -= temp_data.timestamp.values[0]
                            drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
                                    temp_data['direction_z'] == 0)].index
                            if len(drop_index) > 0:
                                loss_indices = set(list(drop_index) + list(drop_index + 1) + list(drop_index + 2))
                                if len(temp_data) in loss_indices:
                                    loss_indices.remove(len(temp_data))
                                if len(temp_data) + 1 in loss_indices:
                                    loss_indices.remove(len(temp_data) + 1)
                                temp_data.loc[loss_indices] = np.nan
                                temp_data = temp_data.interpolate()
                                # temp_data.loc[
                                #     set(list(drop_index) + list(drop_index + 1) + list(drop_index + 2))] = np.nan
                                # temp_data = temp_data.interpolate()
                            temp_data['cursor_speed'] = temp_data.cursor_angular_distance.diff(
                                1) / temp_data.timestamp.diff(1)
                            temp_data['cursor_speed'] = abs(
                                temp_data.cursor_speed.rolling(10, min_periods=1, center=True).mean())
                            validate, reason = validate_trial_data(temp_data)
                            if not validate:  # in case of invalid trial.
                                trial_summary['error'] = reason
                                summary.loc[len(summary)] = trial_summary
                                continue

                                # temp_data = temp_data.drop(
                                #     set(list(drop_index) + list(drop_index + 1) + list(drop_index + 2)))
                            #     print('drop length', len(drop_index), sub_num, pos, cursor_type, rep, t)
                            #     # raise ValueError
                            # temp_data = temp_data.drop(drop_index)
                            # only_success = temp_data[temp_data.target_name == "Target_" + str(t)]
                            only_success = temp_data[temp_data.cursor_angular_distance < default_target_size]
                            if len(only_success) <= 0:
                                raise ValueError('no success frames', len(only_success))

                            initial_contact_time = only_success.timestamp.values[0]
                            # mean required target size
                            mrt_df = temp_data[(temp_data.timestamp >= initial_contact_time) & (
                                    temp_data.timestamp <= dwell_time + initial_contact_time)]
                            mrt = max(list(mrt_df.cursor_angular_distance))
                            trial_summary['required_target_size'] = mrt

                            all_success_dwells = []
                            temp_data['target_in'] = temp_data['cursor_angular_distance'] < default_target_size
                            # for k, g in itertools.groupby(temp_data.iterrows(), key=lambda row: row[1]['target_in']):
                            for k, g in itertools.groupby(temp_data.iterrows(),
                                                          key=lambda row: row[1]['target_name']):
                                # print(k, [t[0] for t in g])
                                # if k == True:
                                if k == 'Target_' + str(t):
                                    df = pd.DataFrame([r[1] for r in g])
                                    all_success_dwells.append(df)
                            success_dwells = []

                            for dw in all_success_dwells:
                                if (dw.timestamp.values[-1] - dw.timestamp.values[0]) > dwell_time - 2 / 60:
                                    success_dwells.append(dw)
                            if len(success_dwells) <= 0:
                                raise ValueError('no success dwell', len(success_dwells))
                            time_sum = 0
                            for dw in success_dwells:
                                time_sum += dw.timestamp.values[-1] - dw.timestamp.values[0]

                            target_in_count = len(success_dwells)
                            target_in_total_time = time_sum
                            target_in_mean_time = time_sum / target_in_count

                            # dwell_temp = temp_data[temp_data.timestamp >= initial_contact_time]
                            """
                            successful trials percentage,
                            time to achieve success
                            number of extra times over 
                            target mean speed during final 100ms
                            mean required-target-size
                            """

                            # time to achieve success
                            first_dwell_time = None
                            for index, dw in enumerate(success_dwells):
                                if index == 0:
                                    first_dwell_time = dw.timestamp.values[0]
                            # mean speed during final 100 ms
                            final_speeds = []
                            for dw in success_dwells:
                                frame = int(dwell_time*60)
                                final_speed = dw.cursor_speed[frame-6:frame].mean()
                                final_speeds.append(final_speed)
                            mean_final_speed = sum(final_speeds) / len(final_speeds)
                            trial_summary = {'dwell_time': dwell_time, 'subject_num': sub_num,
                                             'posture': pos,
                                             'cursor_type': cursor_type,
                                             'repetition': rep,
                                             'target_num': t,
                                             'wide': wide,
                                             'initial_contact_time': initial_contact_time,
                                             'target_in_count': float(target_in_count),
                                             'target_in_total_time': target_in_total_time,
                                             'target_in_mean_time': target_in_mean_time,
                                             'first_dwell_time': first_dwell_time,
                                             'required_target_size': mrt,
                                             'mean_final_speed': mean_final_speed,
                                             'error': None
                                             }
                            summary.loc[len(summary)] = trial_summary

                        except Exception as e:
                            fail_count += 1
                            error_summary = {
                                'dwell_time': dwell_time, 'subject_num': sub_num,
                                'posture': pos,
                                'cursor_type': cursor_type,
                                'repetition': rep,
                                'target_num': t,
                                'wide': wide,
                                'required_target_size': mrt,
                                'error': e.args
                            }
                            summary.loc[len(summary)] = error_summary
                            print(dwell_time, sub_num, pos, cursor_type, rep, t, e.args, 'fail count', fail_count)

    final_summary = summary.groupby([summary['posture'], summary['cursor_type'], summary['wide']]).mean()
    final_summary.to_csv('dwell_time_summary' + str(dwell_time) + '.csv')
    summary.to_csv('dwell_time_Rawsummary' + str(dwell_time) + '.csv')
    return summary


def watch_errors():
    subjects = range(24)
    for sub_num in subjects:
        rep_small = [0, 2, 4, 6, 8]
        rep_large = [1, 3, 5, 7, 9]
        draw_plot = False
        cursorTypes = ['HEAD', 'EYE', 'HAND']

        postures = ['WALK','STAND']
        targets = range(9)
        repetitions = [4, 5, 6, 7, 8, 9]
        # repetitions = range(10)
        fail_count = 0
        summary = pd.DataFrame(
            columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide', 'mean_offset',
                     'std_offset', 'overall_mean_offset', 'overall_std_offset',
                     'initial_contact_time', 'target_in_count', 'target_in_total_time', 'target_in_mean_time',
                     'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal', 'std_offset_vertical',
                     'error'])
        outlier_count = 0
        loss_count = 0
        total_count = 0
        for cursor_type in cursorTypes:
            for rep in repetitions:
                for pos in postures:
                    data = read_hololens_data(sub_num, pos, cursor_type, rep)

                    wide = 'SMALL' if rep in rep_small else 'LARGE'
                    splited_data = split_target(data)
                    # data = data[5:]
                    for t in targets:
                        try:
                            total_count += 1
                            validation, reason = validate_trial_data(splited_data[t])
                            if validation == False:
                                if reason == 'loss':
                                    loss_count += 1
                                elif reason == 'jump':
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
                            print('error', sub_num, pos, cursor_type, rep, e.args)
        print('outlier movement per person', sub_num, 'jump', outlier_count, 'loss', loss_count, total_count)


def plt_confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
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
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

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
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                             [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix

    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
    return path


def find_outliers():
    summary_dataframe = visualize_offsets(show_plot=False)
    head_data = summary_dataframe[
        (summary_dataframe['posture'] == 'WALK') & (summary_dataframe['cursor_type'] == 'HEAD')]
    head_horizontals = []
    for hh in head_data.horizontal.values:
        head_horizontals += list(hh)
    parameters = {}
    for ct, pos, ax in itertools.product(['EYE', 'HAND', 'HEAD'], ['WALK', 'STAND'], ['horizontal', 'vertical']):
        head_data = summary_dataframe[(summary_dataframe['posture'] == pos) & (summary_dataframe['cursor_type'] == ct)]
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
