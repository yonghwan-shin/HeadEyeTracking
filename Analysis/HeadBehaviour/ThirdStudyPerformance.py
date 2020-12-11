# %%
from AnalysingFunctions import *

from FileHandling import *

dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
rolling_size = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def dwell_analysis(target, env, block, subject, output, apply: str):
    walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
    if (walklength < 3.5):
        print('too short walklength');
        return
        # continue;

    if env == 'W':
        r = 0.03 * 5
    else:
        r = 0.03

    output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi

    initial_contact_time_data = output[output[apply] < output['MaximumTargetAngle']]
    if len(initial_contact_time_data) <=0 :
        return;
    initial_contact_time = initial_contact_time_data.timestamp.values[0]

    # print('initial contact:', initial_contact_time)
    Approach = output[output.timestamp <= initial_contact_time].reset_index(drop=True)
    Maintain = output[output.timestamp > initial_contact_time].reset_index(drop=True)
    Maintain['targetOn'] = np.where(Maintain[apply] < Maintain['MaximumTargetAngle'], True, False)
    mean_error = Maintain.TargetHorizontal.mean()
    std_error = Maintain.TargetHorizontal.std()
    # print('mean error:', mean_error, 'std error', std_error)

    dwell_list = []
    maintain_total_frame = Maintain.shape[0]
    target_on_frame = (Maintain[Maintain.targetOn == True]).shape[0]

    for k, l in itertools.groupby(Maintain.iterrows(), key=lambda row: row[1]['targetOn']):
        dwell_list.append(pd.DataFrame([t[1] for t in l]))

    # print('total frame:', maintain_total_frame, 'target-on frame:', target_on_frame, 'target-on rate',
    #       target_on_frame / maintain_total_frame * 100, '%')

    target_in_count = len(dwell_list)
    # print('target in count:', target_in_count)

    if target_in_count > 0:
        total_target_on_time = sum([dwell.timestamp.values[-1] - dwell.timestamp.values[0] for dwell in dwell_list])
        mean_target_on_time = total_target_on_time / target_in_count
    else:  # if there is no target-in
        total_target_on_time = 0
        mean_target_on_time = 0
    result = dict(apply=apply,
                  target=target, environment=env, block=block, subject=subject,
                  walklength=walklength,
                  initial_contact_time=initial_contact_time,
                  mean_error=mean_error, std_error=std_error,
                  maintain_total_frame=maintain_total_frame, target_on_frame=target_on_frame,
                  target_on_rate=target_on_frame / maintain_total_frame * 100,
                  target_in_count=target_in_count, total_target_on_time=total_target_on_time,
                  mean_target_on_time=mean_target_on_time
                  )

    for threshold in dwell_thresholds:
        dwell_success_list = []
        for dwell in dwell_list:
            if len(dwell) * 1 / 60 > threshold:
                dwell_success_list.append(dwell)
        if len(dwell_success_list) <= 0:
            print('no dwell success');
            continue;
        first_dwell_success_time = dwell_success_list[0].timestamp.iloc[0]
        # print('first dwell success', first_dwell_success_time)
        dwell_success_frame = 0
        mean_angular_speed = []

        for success_dwell in dwell_success_list:  # loop through the successive dwells
            mean_angular_speed.append(success_dwell.angle_speed.mean())

            dwell_success_frame += len(success_dwell)
        result['dwell_success_frame_' + str(threshold)] = dwell_success_frame
        result['dwell_success_count_' + str(threshold)] = len(dwell_success_list)
        result['first_dwell_' + str(threshold)] = first_dwell_success_time
        result['mean_angular_speed_' + str(threshold)] = sum(mean_angular_speed) / len(dwell_success_list)
    return result


# %%


# third
subjects = range(1, 2)
envs = ['U', 'W']
targets=[0]
blocks=[1]
# targets = range(8)
# blocks = range(1, 5)
final_result = []

for subject, env, target, block, roll in itertools.product(
        subjects, envs, targets, blocks, rolling_size
):
    try:
        print(subject, env, target, block, roll)
        output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                                    study_num=3)
        output['rolling_average'+str(roll)] = get_new_angular_distance(output.head_rotation_y.rolling(roll, min_periods=1).mean(),
                                                     output.head_rotation_x.rolling(roll, min_periods=1).mean(), output)

        result = dwell_analysis(target, env, block, subject, output, 'rolling_average'+str(roll))

        if result is not None:
            final_result.append(result)
    except Exception as e:
        print(e)
summary = pd.DataFrame(final_result)
summary.to_csv('summary3.csv')
# %%
summary = pd.read_csv('summary3.csv')
UI = summary[summary.environment == 'U']
World = summary[summary.environment == 'W']
rolling_size = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
plot_dwell_columns = ['dwell_success_frame', 'dwell_success_count', 'first_dwell', 'mean_angular_speed']
for col in plot_dwell_columns:
    for roll in rolling_size:
        UI1 = UI[UI['apply'] == roll].fillna(0)
        plt.plot(dwell_thresholds, [UI1[col + "_" + str(x)].mean() for x in dwell_thresholds], label=str(roll))
    plt.title(str(col))
    plt.legend()
    plt.show()
# plt.show()
# for columnName in plot_dwell_columns:
#     draw_walk_plot(UI, World, columnName, dwell_thresholds)


# rolling_size=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
# plot_rolling_columns = ['H_rolling_initial_contact_time', 'H_rolling_mean_error', 'H_rolling_std_error',
#                         'V_rolling_initial_contact_time', 'V_rolling_mean_error', 'V_rolling_std_error']
# for columnName in plot_rolling_columns:
#     # draw_rolling_plot(columnName)
#     draw_walk_plot(UI,World, columnName, rolling_size)
#
# lowpass_cutoff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# plot_lowpass_columns = ['H_lowpass_initial_contact_time', 'H_lowpass_std_error',
#                         'V_lowpass_initial_contact_time', 'V_lowpass_std_error']
# for columnName in plot_lowpass_columns:
#     draw_walk_plot(UI,World,columnName, lowpass_cutoff)
