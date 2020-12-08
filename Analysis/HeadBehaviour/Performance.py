# %%
from FileHandling import *
import matplotlib.pyplot as plt

dwell_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


# %%

def compare_horizontal(data, _output):
    Target_error = data - _output.Phi
    if len(Target_error[abs(Target_error) < 0.1]) <= 0:
        target_in_index = Target_error.abs().idxmin()

    else:
        target_in_index = Target_error[abs(Target_error) < 0.1].index[0]
    # print('target-in time', output.timestamp.iloc[target_in_index], 'mean', Target_error[target_in_index:].mean(),
    #       'std', Target_error[target_in_index:].std())
    return _output.timestamp.iloc[target_in_index], Target_error[target_in_index:].mean(), Target_error[
                                                                                           target_in_index:].std()


# second study: subject= 1~11 , env= U,W , pos = S,W , target= 0~7 , block= 0~5
# third study : subject = 1~16 , env = U,W, pos = None, targets = 0~7, block = 0~5
subjects = range(1, 12)
envs = ["U", "W"]
poss = ['S', 'W']
targets = range(8)
blocks = range(1, 5)

final_result = []
for subject, env, pos, target, block in itertools.product(
        subjects, envs, poss, targets, blocks
):
    try:
        print(subject, env, pos, target, block)
        output = read_hololens_data(target=target, environment=env, posture=pos, block=block, subject=subject,
                                    study_num=2)

        walklength = output.head_position_z.values[-1] - output.head_position_z.values[0]
        if (walklength < 3.5) and (pos == 'W'):
            print('too short walklength');
            continue;
        if env == 'W':
            r = 0.05 * 5
        else:
            r = 0.05
        output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
        initial_contact_time = output[output['angular_distance'] < output['MaximumTargetAngle']].timestamp.values[0]

        # print('initial contact:', initial_contact_time)
        Approach = output[output.timestamp <= initial_contact_time].reset_index(drop=True)
        Maintain = output[output.timestamp > initial_contact_time].reset_index(drop=True)
        Maintain['targetOn'] = np.where(Maintain['angular_distance'] < Maintain['MaximumTargetAngle'], True, False)
        mean_error = Maintain.TargetHorizontal.mean()
        std_error = Maintain.TargetHorizontal.std()
        # print('mean error:', mean_error, 'std error', std_error)
        dwell_list = []
        maintain_total_frame = Maintain.shape[0]
        target_on_frame = (Maintain[Maintain.targetOn == True]).shape[0]

        # for k, g in itertools.groupby(Maintain.targetOn):  # search consecutive True (target-on) data
        #     if k:  # if target-on
        #         g = list(g)
        #         dwell_list.append(g)
        for k, l in itertools.groupby(Maintain.iterrows(), key=lambda row: row[1]['targetOn']):
            dwell_list.append(pd.DataFrame([t[1] for t in l]))

        # print('total frame:', maintain_total_frame, 'target-on frame:', target_on_frame, 'target-on rate',
        #       target_on_frame / maintain_total_frame * 100, '%')
        target_in_count = len(dwell_list)
        # print('target in count:', target_in_count)

        result = dict(
            target=target, environment=env, posture=pos, block=block, subject=subject,
            walklength=walklength,
            initial_contact_time=initial_contact_time,
            mean_error=mean_error, std_error=std_error,
            maintain_total_frame=maintain_total_frame, target_on_frame=target_on_frame,
            target_on_rate=target_on_frame / maintain_total_frame * 100,
            target_in_count=target_in_count,
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

            dwell_success_frame = 0
            mean_angular_speed = []

            for success_dwell in dwell_success_list:  # loop through the successive dwells
                mean_angular_speed.append(success_dwell.angle_speed.mean())

                dwell_success_frame += len(success_dwell)
            result['dwell_success_frame_' + str(threshold)] = dwell_success_frame
            result['dwell_success_count_' + str(threshold)] = len(dwell_success_list)
            result['first_dwell_' + str(threshold)] = first_dwell_success_time
            result['mean_angular_speed_' + str(threshold)] = sum(mean_angular_speed) / len(dwell_success_list)
            # print('threshold', threshold, 'dwell success frame', dwell_success_frame,
            #       'dwell success count', len(dwell_success_list), 'first dwell', first_dwell_success_time,
            #       'mean angular speed',
            #       sum(mean_angular_speed) / len(dwell_success_list))

        # def rolling_horizontal(period):
        #     Target_error = output.head_rotation_y.rolling(period, min_periods=1).mean() - output.Phi
        #     target_in_index = Target_error[abs(Target_error) < 0.1].index[0]
        #     print('rolling', period, 'target-in', output.timestamp.iloc[target_in_index], 'mean',
        #           Target_error[target_in_index:].mean(), 'std', Target_error[target_in_index:].std())
        #     return Target_error

        rolling_size = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        for window in rolling_size:
            t, m, sd = compare_horizontal(output.head_rotation_y.rolling(window, min_periods=1).mean(), output)
            result['rolling_initial_contact_time_' + str(window)] = t
            result['rolling_mean_error_' + str(window)] = m
            result['rolling_std_error_' + str(window)] = sd

        final_result.append(result)
    except Exception as e:
        print(e)
summary = pd.DataFrame(final_result)
# plt.plot(output.timestamp, output.TargetHorizontal)
# # plt.plot(output.timestamp,output.Phi)
# plt.plot(output.timestamp, rolling_horizontal(30))
# plt.plot(output.timestamp, rolling_horizontal(60))
# plt.plot(output.timestamp, rolling_horizontal(90))
# plt.plot(output.timestamp, rolling_horizontal(120))
# plt.show()

# %%
stand = summary[summary.posture == 'S']
walk = summary[summary.posture == 'W']
