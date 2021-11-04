"""
cursor_angular_distance', 'start_num', 'end_num', 'timestamp',
'target_position_x', 'target_position_y', 'target_position_z',
'head_position_x', 'head_position_y', 'head_position_z',
'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
'head_forward_x', 'head_forward_y', 'head_forward_z', 'cursor_type',
'target_name', 'origin_x', 'origin_y', 'origin_z', 'direction_x',
'direction_y', 'direction_z', 'ray_origin_x', 'ray_origin_y',
'ray_origin_z', 'ray_direction_x', 'ray_direction_y',
'ray_direction_z']
"""
# %%
import itertools

import pandas as pd

from FileHandling import *
import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)  # <==== 경고를 끈다

rep_small = [0, 2, 4, 6, 8]
rep_large = [1, 3, 5, 7, 9]
draw_plot = False
cursorTypes = ['HEAD', 'EYE', 'HAND']

postures = ['STAND', 'WALK']
targets = range(9)
repetitions = [4, 5, 6, 7, 8, 9]
# repetitions = [5]
# cursorTypes = ['HAND']
sub_num = 0

#%%
df = get_one_trial(0,'STAND','HEAD',6,4)
#%%
# summary = pd.DataFrame(columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num'])
def summarize_subject(sub_num):
    fail_count = 0
    summary = pd.DataFrame(
        columns=['subject_num', 'posture', 'cursor_type', 'repetition', 'target_num', 'wide', 'mean_offset',
                 'std_offset',
                 'initial_contact_time', 'target_in_count', 'target_in_total_time', 'target_in_mean_time',
                 'mean_offset_horizontal', 'mean_offset_vertical', 'std_offset_horizontal', 'std_offset_vertical'])
    for cursor_type in cursorTypes:
        for rep in repetitions:
            for pos in postures:
                data = read_hololens_data(sub_num, pos, cursor_type, rep)
                splited_data = split_target(data)
                wide = 'SMALL' if rep in rep_small else 'LARGE'
                for t in targets:
                    try:
                        temp_data = splited_data[t]
                        temp_data.reset_index(inplace=True)
                        temp_data.timestamp -= temp_data.timestamp.values[0]
                        drop_index = temp_data[(temp_data['direction_x'] == 0) & (temp_data['direction_y'] == 0) & (
                                temp_data['direction_z'] == 0)].index
                        # if len(drop_index) >0: print('drop length',len(drop_index),sub_num, pos, cursor_type, rep, t)
                        temp_data = temp_data.drop(drop_index)
                        initial_contact_time = temp_data[temp_data.target_name == "Target_" + str(t)].timestamp.values[
                            0]
                        dwell_temp = temp_data[temp_data.timestamp > initial_contact_time]
                        success_dwells = []
                        for k, g in itertools.groupby(dwell_temp.iterrows(), key=lambda row: row[1]['target_name']):
                            # print(k, [t[0] for t in g])
                            if k == 'Target_' + str(t):
                                df = pd.DataFrame([r[1] for r in g])
                                success_dwells.append(df)
                        time_sum = 0
                        for dw in success_dwells:
                            time_sum += dw.timestamp.values[-1] - dw.timestamp.values[0]

                        target_in_count = len(success_dwells)
                        target_in_total_time = time_sum
                        target_in_mean_time = time_sum / target_in_count
                        # horizontal / vertical
                        import numpy as np
                        import numpy.linalg as LA
                        # def angle_between(a, b):
                        #     inner = np.inner(a, b)
                        #     norms = LA.norm(a) * LA.norm(b)
                        #     cos = inner / norms
                        #     rad = np.arccos(np.clip(cos, -1.0, 1.0))
                        #     deg = np.rad2deg(rad)
                        #     return deg

                        # print(temp_data)
                        dwell_temp['cursor_rotation'] = dwell_temp.apply(
                            lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z), axis=1)
                        dwell_temp['target_rotation'] = dwell_temp.apply(
                            lambda x: asSpherical(x.target_position_x - x.origin_x, x.target_position_y - x.origin_y,
                                                  x.target_position_z - x.origin_z), axis=1)
                        dwell_temp['offset_horizontal'] = dwell_temp.apply(
                            lambda x: x.cursor_rotation[1] - x.target_rotation[1], axis=1)
                        dwell_temp['offset_vertical'] = dwell_temp.apply(
                            lambda x: x.cursor_rotation[0] - x.target_rotation[0], axis=1)

                        mean_offset_horizontal = dwell_temp.offset_horizontal.mean()
                        std_offset_horizontal = dwell_temp.offset_horizontal.std()
                        mean_offset_vertical = dwell_temp.offset_vertical.mean()
                        std_offset_vertical = dwell_temp.offset_vertical.std()

                        summary.loc[len(summary)] = {'subject_num': sub_num,
                                                     'posture': pos,
                                                     'cursor_type': cursor_type,
                                                     'repetition': rep,
                                                     'target_num': t,
                                                     'wide': wide,
                                                     'mean_offset': dwell_temp.cursor_angular_distance.mean(),
                                                     'std_offset': dwell_temp.cursor_angular_distance.std(),
                                                     'initial_contact_time': initial_contact_time,
                                                     'target_in_count': float(target_in_count),
                                                     'target_in_total_time': target_in_total_time,
                                                     'target_in_mean_time': target_in_mean_time,
                                                     'mean_offset_horizontal': mean_offset_horizontal,
                                                     'mean_offset_vertical': mean_offset_vertical,
                                                     'std_offset_horizontal': std_offset_horizontal,
                                                     'std_offset_vertical': std_offset_vertical
                                                     }
                        # smalls.loc[len(smalls)] = [sub_num, 'STAND', cursor_type, rep, t, temp_data.cursor_angular_distance.mean(),
                        #                            temp_data.cursor_angular_distance.std()]
                    except Exception as e:
                        fail_count += 1
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
    final_summary.to_csv('summary' + str(sub_num) + '.csv')
    summary.to_csv('Rawsummary' + str(sub_num) + '.csv')
    return summary


# %%
# subjects= [0,1,2,3,4,5,6,7,9]
subjects = [0]
for subject in subjects:
    a = summarize_subject(subject)

# %%
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

dfs = []
for subject in subjects:
    summary = pd.read_csv("summary" + str(subject) + ".csv")
    dfs.append(summary)
    # fig = px.bar(summary, x='cursor_type', y=['mean_offset', 'std_offset', 'initial_contact_time'], barmode='group',
    #              facet_row='posture', facet_col='wide', title='summary '+str(subject))
    #
    # fig.show()
df_average = pd.concat(dfs)
fs = df_average.groupby([df_average['posture'], df_average['cursor_type'], df_average['wide']]).mean()
fs.to_csv("total_summary.csv")
fs = pd.read_csv('total_summary.csv')
fig = px.bar(fs, x='cursor_type', y=['mean_offset', 'std_offset', 'initial_contact_time'], barmode='group',
             facet_row='posture', facet_col='wide', title='total summary')

fig.show()
# %%


# %%
sub_num = 4
rep_small = [0, 2, 4, 6, 8]
rep_large = [1, 3, 5, 7, 9]
draw_plot = False
# cursorTypes = ['HEAD', 'EYE', 'HAND']
cursorTypes = ['EYE']
postures = ['STAND', 'WALK']
# targets = range(9)
targets = range(9)
repetitions = range(9)
for cursor_type in cursorTypes:
    total_stand_offset_means = []
    total_stand_offset_stds = []
    total_walk_offset_means = []
    total_walk_offset_stds = []
    for rep in repetitions:

        data = read_hololens_data(sub_num, 'STAND', cursor_type, rep)
        walkdata = read_hololens_data(sub_num, 'WALK', cursor_type, rep)
        # print(data.columns)
        splited_data = split_target(data)
        walk_splited_data = split_target(walkdata)

        stand_offset_means = []
        stand_offset_stds = []
        walk_offset_means = []
        walk_offset_stds = []

        for i in targets:
            try:
                temp = splited_data[i]
                walk_temp = walk_splited_data[i]
                temp.reset_index(inplace=True)
                temp.timestamp -= temp.timestamp.values[0]
                walk_temp.reset_index(inplace=True)
                walk_temp.timestamp -= walk_temp.timestamp.values[0]
                initial_contact_time = temp[temp.target_name == "Target_" + str(i)].timestamp.values[0]
                walk_initial_contact_time = walk_temp[walk_temp.target_name == "Target_" + str(i)].timestamp.values[0]

                dwell_temp = temp[temp.timestamp > initial_contact_time]
                offset_mean = dwell_temp['cursor_angular_distance'].mean()
                offset_std = dwell_temp['cursor_angular_distance'].std()
                walk_dwell_temp = walk_temp[walk_temp.timestamp > walk_initial_contact_time]
                walk_offset_mean = walk_dwell_temp['cursor_angular_distance'].mean()
                walk_offset_std = walk_dwell_temp['cursor_angular_distance'].std()
                stand_offset_means.append(offset_mean)
                stand_offset_stds.append(offset_std)
                walk_offset_means.append(walk_offset_mean)
                walk_offset_stds.append(walk_offset_std)
                # print(i,'stand',offset_mean,offset_std)
                # print(i, 'walk', walk_offset_mean, walk_offset_std)
                fail = len(dwell_temp[dwell_temp.isEyeTrackingDataValid == False])
                if fail != 0:
                    print(fail)

                if (draw_plot):
                    plt.plot(temp.timestamp, temp.cursor_angular_distance, label='stand')
                    plt.plot(walk_temp.timestamp, walk_temp.cursor_angular_distance, label='walk')
                    plt.axvline(initial_contact_time)
                    plt.axvline(walk_initial_contact_time)
                    # plt.text(initial_contact_time,0,'initial contact time',)
                    plt.legend()
                    plt.title(f'{cursor_type} angular offset from :target' + str(i))
                    plt.show()
            # print(cursor_type,rep,'\n','stand offset mean',sum(stand_offset_means)/len(stand_offset_means),
            #       'stand offset std',sum(stand_offset_stds)/len(stand_offset_stds),'\n',
            #       'walk offset mean', sum(walk_offset_means) / len(walk_offset_means),
            #       'walk offset std', sum(walk_offset_stds) / len(walk_offset_stds)
            #       )
            except Exception as e:
                print(e)
        total_stand_offset_means.append(sum(stand_offset_means) / len(stand_offset_means))
        total_stand_offset_stds.append(sum(stand_offset_stds) / len(stand_offset_stds))
        total_walk_offset_means.append(sum(walk_offset_means) / len(walk_offset_means))
        total_walk_offset_stds.append(sum(walk_offset_stds) / len(walk_offset_stds))
    print('subject', sub_num, cursor_type, '\n', 'stand offset mean',
          sum(total_stand_offset_means) / len(total_stand_offset_means),
          'stand offset std', sum(total_stand_offset_stds) / len(total_stand_offset_stds), '\n',
          'walk offset mean', sum(total_walk_offset_means) / len(total_walk_offset_means),
          'walk offset std', sum(total_walk_offset_stds) / len(total_walk_offset_stds)
          )
