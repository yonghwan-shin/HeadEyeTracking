# %% Importing
import itertools

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# custom functions
from FileHandling import get_one_subject_files, get_file_by_info, file_as_pandas
from DataManipulation import check_holo_file

subjects = range(
    201, 211
)  # Subject No1. To change subject groups for whole subjects, use -> range(201,211)
targets = range(8)
envs = ["U", "W"]
poss = ["W", "S"]
blocks = range(1, 5)


def see_holo(postures: list, environments: list):
    startpositionX = []
    startAngular = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(
            subject, refined=True
        )
        for target, env, pos, block in itertools.product(
                targets, environments, postures, blocks
        ):
            current_info = [target, env, pos, block]
            try:
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_holo = file_as_pandas(hololens_data)

                check_holo_file(df_holo, current_info)  # Check the trial data is fine

                target_name = "target_" + str(target)
                first = df_holo.head(1)
                startX = first['HeadPositionX'].values[0]
                startAngle = first['TargetAngularDistance'].values[0]
                startpositionX.append(startX)
                startAngular.append(startAngle)
            except Exception as e:
                print(e.args)
    return pd.Series(startpositionX), pd.Series(startAngular)


def get_dwell(postures: list):
    dwell_time_list = []
    total_dwell_list = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(
            subject, refined=True
        )
        for target, env, pos, block in itertools.product(
                targets, envs, postures, blocks
        ):
            current_info = [target, env, pos, block]
            # Search each trial here
            try:
                # Preparing dataset
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_holo = file_as_pandas(hololens_data)

                check_holo_file(df_holo, current_info)  # Check the trial data is fine

                target_name = "target_" + str(target)
                _init_time = df_holo.head(1)["Timestamp"].values[0]
                _end_time = df_holo.tail(1)["Timestamp"].values[0]
                total_time = _end_time - _init_time
                frame_rate = df_holo.shape[0] / total_time

                list_target_in = []
                list_target_out = []
                list_duration_in = []
                list_duration_out = []
                previous_target = ""

                # Search when is 'target-in' and 'target-out'
                for row in df_holo.itertuples(index=False):
                    if row[13] == target_name and previous_target != target_name:
                        list_target_in.append(row[0])
                    if row[13] != target_name and previous_target == target_name:
                        list_target_out.append(row[0])
                    previous_target = row[13]

                num_over = len(list_target_in)
                num_off = len(list_target_out)
                if num_over < 1:  # if there is no encounter...?
                    print(current_info, "zero over count: never points the target")
                    continue

                start_time = list_target_in[0] - _init_time
                if num_over == num_off:
                    for in_count in range(num_off):
                        list_duration_in.append(
                            list_target_out[in_count] - list_target_in[in_count]
                        )
                elif num_over == num_off + 1:
                    for in_count in range(num_off):
                        list_duration_in.append(
                            list_target_out[in_count] - list_target_in[in_count]
                        )
                    list_duration_in.append(_end_time - list_target_in[-1])
                else:
                    print("error in matching in n out")
                    continue
                dur_over = np.sum(list_duration_in)
                total_dwell_list.append(dur_over)
                dwell_time_list.append(list_duration_in)
            except ValueError as err:
                print(subject, current_info, err)
    return dwell_time_list, total_dwell_list


# %%
#
# outputList_w, total_w = get_dwell(["W"])
# outputList_s, total_s = get_dwell(["S"])
#
# merged_w = list(itertools.chain(*outputList_w))
# merged_s = list(itertools.chain(*outputList_s))

# %% counting every target-in


# dwells_w = np.array(merged_w)
# dwells_s = np.array(merged_s)
# print(dwells_w.mean(), dwells_s.mean())


def plot_dwell(dwells, axis):
    x = []
    y = []
    for i in np.arange(100, 1510, step=100):
        i = i / 1000
        # Check how many dwells have longer than certain threshold
        print(
            i,
            ":  ",
            len(dwells[dwells > i]),
            len(dwells[dwells > i]) / len(dwells) * 100,
            "%",
        )
        x.append(i)  # threshold
        y.append(len(dwells[dwells > i]) / len(dwells) * 100)  # Percentages
    axis.plot(x, y)
    # plt.show()


# fig, axis = plt.subplots(1, 2, sharey=True)
# print("-" * 20, "WALK", "-" * 20)
# plot_dwell(dwells_w, axis[0])
#
# print("-" * 20, "STAND", "-" * 20)
# plot_dwell(dwells_s, axis[1])
#
# plt.show()


# %% Success/Fail in trial


def find_dwell(outputList, threshold):
    count = 0
    for trial in outputList:
        trial = np.array(trial)
        if len(trial[trial > threshold]) > 0:
            count = count + 1
    return count / len(outputList)


# fig, axis = plt.subplots()
# for th in np.arange(100, 1510, step=100):
#     th = th / 1000
#     w = find_dwell(outputList_w, th)
#     if th < 0.11:
#         axis.bar(th, w * 100, color="blue", width=0.03, alpha=0.7, label="walk")
#     else:
#         axis.bar(th, w * 100, color="blue", width=0.03, alpha=0.7)
#     print(th, w)
#
# for th in np.arange(100, 1510, step=100):
#     th = th / 1000
#     w = find_dwell(outputList_s, th)
#     if th < 0.11:
#         axis.bar(
#             th, w * 100, color="red", width=0.03, alpha=0.7, align="edge", label="stand"
#         )
#     else:
#         axis.bar(th, w * 100, color="red", width=0.03, alpha=0.7, align="edge")
#     print(th, w)
# axis.set_xlabel("threshold (sec)")
# axis.set_ylabel("success rate (%)")
# axis.set_title("Dwell Success rate")
# # fig, ax = plt.subplots()
# # rects_w = ax.bar()
# plt.legend()
# plt.show()

# %% See what's the problem in standing condition...
# subject = 202
# target = 2
# env = "W"
# pos = "S"
# block = 1
# [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(
#     subject, refined=True
# )
# current_info = [target, env, pos, block]
# hololens_data = get_file_by_info(hololens_file_list, current_info)
# df_holo = file_as_pandas(hololens_data)
# check_holo_file(df_holo, current_info)
# target_name = "target_" + str(target)
# plt.plot(df_holo.Timestamp, df_holo.TargetAngularDistance)
# # Draw vertical line in 'successful pointing'
# for target_in in (df_holo[df_holo.TargetEntered == target_name]).Timestamp:
#     plt.axvline(target_in, alpha=0.2)
# plt.show()
# %%
# subjects = range(
#     201, 212
# )  # Subject No1. To change subject groups for whole subjects, use -> range(201,211)
# targets = range(8)
# envs = ["U", "W"]
# poss = ["W", "S"]
# blocks = range(1, 5)
# x_sw, a_sw = see_holo('S', 'W')
# x_su, a_su = see_holo('S', 'U')
# x_ww, a_ww = see_holo('W', 'W')
# x_wu, a_wu = see_holo('W', 'U')
# x_w,a_w = see_holo('W')

# %%
from scipy.stats import ks_2samp


#
# for comb in itertools.permutations([x_sw, a_sw, x_su, a_su, x_ww, a_ww, x_wu, a_wu], r=2):
#     print(ks_2samp(comb[0], comb[1]))


# %%

def angle_velocity(x1, y1, z1, x2, y2, z2, _time):
    import vg
    # if type(_head_forward2) is not dict: return None
    vector1 = np.array([x1, y1, z1])
    vector2 = np.array([x2, y2, z2])
    return vg.angle(vector1, vector2) / _time


def summary(subject, env, target, pos, block, threshold):
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(
        subject, refined=True
    )
    current_info = [target, env, pos, block]
    print(subject,current_info, threshold)
    hololens_data = get_file_by_info(hololens_file_list, current_info)
    df_holo = file_as_pandas(hololens_data)
    for col in ['HeadForwardX', 'HeadForwardY', 'HeadForwardZ']:
        df_holo[str(col) + '_next'] = df_holo[col].shift(1)
    df_holo['time_interval'] = df_holo.Timestamp.diff()
    df_holo['angle_speed'] = df_holo.apply(lambda x: angle_velocity(x.HeadForwardX, x.HeadForwardY, x.HeadForwardZ,
                                                                    x.HeadForwardX_next, x.HeadForwardY_next,
                                                                    x.HeadForwardZ_next,
                                                                    x.time_interval), axis=1)
    df_holo['TargetEntered'] = np.where(df_holo['TargetEntered'] == 'target_' + str(target), True, False)
    success = pd.DataFrame(df_holo.TargetEntered, )
    success['next_frame'] = success.TargetEntered.shift(1)
    success['start'] = (success.TargetEntered == True) & (success.next_frame == False)
    success['end'] = (success.TargetEntered == False) & (success.next_frame == True)

    starts = np.array(success.start[success.start == True].index)
    ends = np.array(success.end[success.end == True].index)
    _no_over = True if len(starts) == 0 else False
    if _no_over == True:
        # print('no over')
        return dict(subject=subject,
                    environment=env,
                    posture=pos,
                    target=target,
                    block=block,
                    threshold=threshold,
                    no_over=_no_over, )

        # continue;  # check there is an over-target
    if len(starts) == 1 and len(ends) == 0:
        # print('full dwell')

        ends = np.array([success.index[-1]])
    else:
        if starts[-1] > ends[-1]:  starts = np.delete(starts, -1)
        if starts[0] > ends[0]: ends = np.delete(ends, -1)
    durations = ends - starts
    freq = 60
    dwell = np.argwhere(durations > threshold * freq)
    dwell = [x[0] for x in dwell]
    if len(dwell) <= 0:
        # print('no dwell')
        return dict(subject=subject,
                    environment=env,
                    posture=pos,
                    target=target,
                    block=block,
                    threshold=threshold,
                    dwell_success=False)

        # continue;
    _start_time = df_holo.Timestamp[starts[0]]  # When is the first target-over
    _num_over = len(starts)  # How many times of target-over
    duration_times = [df_holo.Timestamp[ends[i]] - df_holo.Timestamp[starts[i]] for i in range(len(starts))]
    _total_dur_over = sum(duration_times)
    _mean_dur_over = _total_dur_over / _num_over
    _angle_dist_mean = df_holo.TargetAngularDistance.mean()
    _longest_over = max(duration_times)
    _dwell_success = True if len(dwell) > 0 else False  # Success rate
    _dwell_time = df_holo.Timestamp[starts[dwell[0]]]  # first dwell timestamp
    # SPEED
    _mean_speeds = []
    _vel_ins = []
    _vel_outs = []
    _last100s = []
    for i in range(len(dwell)):
        _mean_speed = df_holo.iloc[starts[dwell[i]]:ends[dwell[i]]].angle_speed.mean()
        _vel_in = df_holo.iloc[starts[dwell[i]]].angle_speed
        _vel_out = df_holo.iloc[ends[dwell[i]]].angle_speed
        _last100 = df_holo.iloc[ends[dwell[i]] - 6:ends[dwell[i]]].angle_speed.mean()
        _mean_speeds.append(_mean_speed)
        _vel_ins.append(_vel_in)
        _vel_outs.append(_vel_out)
        _last100s.append(_last100)

    _prior_count = int(dwell[0])

    output = dict(
        subject=subject,
        environment=env,
        posture=pos,
        target=target,
        block=block,
        threshold=threshold,
        no_over=_no_over,
        start_time=_start_time,
        num_over=_num_over,
        total_dur_over=_total_dur_over,
        mean_dur_over=_mean_dur_over,
        angle_dist_mean=_angle_dist_mean,
        longest_over=_longest_over,
        dwell_success=_dwell_success,
        dwell_time=_dwell_time,
        mean_speeds=_mean_speeds,
        vel_ins=_vel_ins,
        vel_outs=_vel_outs,
        last_100s=_last100s,
        prior_count=_prior_count
    )
    return output


if __name__ == "__main__":
    # subject = 201
    # env = 'U'
    # target = 3
    # pos = 'W'
    # block = 2
    # threshold = 0.3
    # dwell_output = []

    subjects = range(201, 212)
    envs = ['U', 'W']
    targets = range(8)
    poss = ['W', 'S']
    blocks = range(1, 5)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    dwell_output = []
    for subject, env, target, pos, block, threshold in itertools.product(subjects, envs, targets, poss, blocks,
                                                                         thresholds):
        try:
            dwell_output.append(summary(subject, env, target, pos, block, threshold))
        except Exception as e:
            print(e.args)

summary = pd.DataFrame(dwell_output)

#%%
summary_sub = dict([])
for subject, env,pos in itertools.product(subjects, envs,poss):
    summary_sub[(subject, env,pos)] = summary.loc[(summary['subject'] == subject) & (summary['environment'] == env)& (summary['posture'] == pos)]

def list_to_mean(data):
    if type(data) is not list:
        return None
    return sum(data) / len(data)
#%%
summary_proceed = []
for subject, env ,pos in itertools.product(subjects, envs,poss ):
    data = summary_sub[(subject, env,pos)]
    _mean_start_time = data[data['threshold'] == 0.1].start_time.mean()
    _mean_num_over = data[data['threshold'] == 0.1].num_over.mean()
    _mean_total_dur_over = data[data['threshold'] == 0.1].total_dur_over.mean()
    _mean_dur_over = data[data['threshold'] == 0.1].mean_dur_over.mean()
    _mean_angle_dist = data[data['threshold'] == 0.1].angle_dist_mean.mean()
    _mean_longest_over = data[data['threshold'] == 0.1].longest_over.mean()
    # no over is  None
    dwell_success_count = dict([])
    dwell_time = dict([])
    mean_speeds = dict([])
    prior_counts = dict([])
    last_100s = dict([])
    for threshold in thresholds:
        dwell_success_count['dwell_success_count' + str(threshold)] = \
            data[(data['threshold'] == threshold) & (data['dwell_success'] == True)].shape[0]
        dwell_time['dwell_time' + str(threshold)] = data[data['threshold'] == threshold].dwell_time.mean()
        mean_speeds['mean_speeds' + str(threshold)] = data[data['threshold'] == threshold].mean_speeds.apply(
            list_to_mean).mean()
        prior_counts['prior_counts' + str(threshold)] = data[data['threshold'] == threshold].prior_count.mean()
        last_100s['last_100s' + str(threshold)] = data[data['threshold'] == threshold].last_100s.apply(
            list_to_mean).mean()
    output = dict(
        subject=subject,
        environment=env,
        posture = pos,
        mean_start_time=_mean_start_time,
        mean_num_over=_mean_num_over,
        mean_total_dur_over=_mean_total_dur_over,
        mean_dur_over=_mean_dur_over,
        mean_angle_dist=_mean_angle_dist,
        mean_longest_over=_mean_longest_over,
        dwell_success_count=dwell_success_count,
        dwell_time=dwell_time,
        mean_speeds=mean_speeds,
        prior_counts=prior_counts,
        last_100s=last_100s
    )
    summary_proceed.append(output)
df_summary_proceed = pd.DataFrame(summary_proceed)
for col in ['dwell_success_count', 'dwell_time', 'mean_speeds', 'prior_counts', 'last_100s']:
    df_summary_proceed = pd.concat([df_summary_proceed.drop([col], axis=1),
                                    df_summary_proceed[col].apply(pd.Series)], axis=1)

#%%
df_summary_proceed.to_excel('proceed.xlsx')