import itertools
from matplotlib import pyplot as plt
import numpy as np
# custom functions
from FileHandling import get_one_subject_files, get_file_by_info, file_as_pandas
from DataManipulation import check_holo_file

subjects = range(201, 202)  # Subject No1. To change subject groups for whole subjects, use -> range(201,211)
targets = range(8)
envs = ['U', 'W']
poss = ['W', 'S']
blocks = range(5)


def get_dwell(postures: list):
    dwell_time_list = []
    total_dwell_list = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, postures, blocks):
            current_info = [target, env, pos, block]
            # Search each trial here
            try:
                # Preparing dataset
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_holo = file_as_pandas(hololens_data)

                check_holo_file(df_holo, current_info)  # Check the trial data is fine

                target_name = 'target_' + str(target)
                _init_time = df_holo.head(1)['Timestamp'].values[0]
                _end_time = df_holo.tail(1)['Timestamp'].values[0]
                total_time = _end_time - _init_time
                frame_rate = df_holo.shape[0] / total_time

                list_target_in = []
                list_target_out = []
                list_duration_in = []
                list_duration_out = []
                previous_target = ''

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
                    print(current_info, 'zero over count: never points the target')
                    continue

                start_time = list_target_in[0] - _init_time
                if num_over == num_off:
                    for in_count in range(num_off):
                        list_duration_in.append(list_target_out[in_count] - list_target_in[in_count])
                elif num_over == num_off + 1:
                    for in_count in range(num_off):
                        list_duration_in.append(list_target_out[in_count] - list_target_in[in_count])
                    list_duration_in.append(_end_time - list_target_in[-1])
                else:
                    print('error in matching in n out')
                    continue
                dur_over = np.sum(list_duration_in)
                total_dwell_list.append(dur_over)
                dwell_time_list.append(list_duration_in)
            except ValueError as err:
                print(subject, current_info, err)
    return dwell_time_list, total_dwell_list


outputList_w, total_w = get_dwell(['W'])
outputList_s, total_s = get_dwell(['S'])

merged_w = list(itertools.chain(*outputList_w))
merged_s = list(itertools.chain(*outputList_s))

# %% counting every target-in


dwells_w = np.array(merged_w)
dwells_s = np.array(merged_s)
print(dwells_w.mean(), dwells_s.mean())


def plot_dwell(dwells, axis):
    x = []
    y = []
    for i in np.arange(100, 1510, step=100):
        i = i / 1000
        # Check how many dwells have longer than certain threshold
        print(i, ':  ', len(dwells[dwells > i]), len(dwells[dwells > i]) / len(dwells) * 100, '%')
        x.append(i)  # threshold
        y.append(len(dwells[dwells > i]) / len(dwells) * 100)  # Percentages
    axis.plot(x, y)
    # plt.show()


fig, axis = plt.subplots(1, 2, sharey=True)
print('-' * 20, 'WALK', '-' * 20)
plot_dwell(dwells_w, axis[0])

print('-' * 20, 'STAND', '-' * 20)
plot_dwell(dwells_s, axis[1])

plt.show()


# %% Success/Fail in trial

def find_dwell(outputList, threshold):
    count = 0
    for trial in outputList:
        trial = np.array(trial)
        if len(trial[trial > threshold]) > 0:
            count = count + 1
    return count / len(outputList)


fig, axis = plt.subplots(1, 2, sharey=True)
for th in np.arange(100, 1510, step=100):
    th = th / 1000
    w = find_dwell(outputList_w, th)
    axis[0].scatter(th, w)
    print(th, w)

for th in np.arange(100, 1510, step=100):
    th = th / 1000
    w = find_dwell(outputList_s, th)
    axis[1].scatter(th, w)
    print(th, w)

plt.show()

# %% See what's the problem in standing condition...
subject = 202
target = 2
env = 'W'
pos = 'S'
block = 1
[imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
current_info = [target, env, pos, block]
hololens_data = get_file_by_info(hololens_file_list, current_info)
df_holo = file_as_pandas(hololens_data)
check_holo_file(df_holo, current_info)
target_name = 'target_' + str(target)
plt.plot(df_holo.Timestamp, df_holo.TargetAngularDistance)
# Draw vertical line in 'successful pointing'
for target_in in (df_holo[df_holo.TargetEntered == target_name]).Timestamp:
    plt.axvline(target_in, alpha=0.2)
plt.show()
