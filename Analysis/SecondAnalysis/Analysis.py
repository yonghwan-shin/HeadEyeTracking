from FileHandling import *
from DataManipulation import *
from Analysing_functions import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import itertools
import seaborn as sns
from scipy import interpolate
from scipy import stats
import numpy as np
import statistics

# subjects = range(201, 216)
subjects = range(201, 212)
# subjects=range(201,202)
targets = range(8)
envs = ['U', 'W']
# envs = ['U']
# poss = ['S', 'W']
poss = ['S']
# blocks=[0]
blocks = range(5)


# eye columns
# circle_3d	confidence	timestamp	diameter_3d	ellipse	location	diameter	sphere	projected_sphere	model_confidence	model_id	model_birth_timestamp	theta	phi	norm_pos	topic	id	method	norm_x	norm_y
# holo columns
# Timestamp	HeadPositionX	HeadPositionY	HeadPositionZ	HeadRotationX	HeadRotationY	HeadRotationZ	HeadForwardX	HeadForwardY	HeadForwardZ	TargetPositionX	TargetPositionY	TargetPositionZ	TargetEntered	TargetAngularDistance
def create_eye_csv():
    # To reduce parsing json time...
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye_data = get_file_by_info(eye_file_list, current_info)
                df_eye = manipulate_eye(file_as_pandas(eye_data))

                norm_x = []
                norm_y = []
                for row in df_eye.itertuples():
                    norm_x.append(float(row[15][0]))
                    norm_y.append(float(row[15][1]))
                df_eye['norm_x'] = norm_x
                df_eye['norm_y'] = norm_y

                print(df_eye.head(1)['norm_x'])
                df_eye.to_csv(path_or_buf=(DATA_ROOT / 'refined_eye_data' / ('refined_' + eye_data.name)), index=False)
                print('saved', eye_data.name, '  ', current_info, subject)
            except ValueError as err:
                print(err, current_info)


def summary_holo_data_proceed():
    df_holo = pd.read_csv(DATA_ROOT / 'hololens_analysis' / 'summary_1st_analysis.csv')
    output = []
    df_output = pd.DataFrame(columns=df_holo.columns)
    df_WW = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'W')]
    df_UW = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'W')]
    df_US = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'S')]
    df_WS = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'S')]
    WW = df_WW.mean()
    WW['env'] = 'W'
    WW['pos'] = 'W'
    UW = df_UW.mean()
    UW['env'] = 'U'
    UW['pos'] = 'W'
    US = df_US.mean()
    US['env'] = 'U'
    US['pos'] = 'S'
    WS = df_WS.mean()
    WS['env'] = 'W'
    WS['pos'] = 'S'

    std_WW = df_WW.std()

    std_UW = df_UW.std()
    std_US = df_US.std()
    std_WS = df_WS.std()

    std_WW['subject'] = ''
    std_WW['target'] = ''
    std_WW['env'] = ''
    std_WW['pos'] = 'SD:'
    std_UW['subject'] = ''
    std_UW['target'] = ''
    std_UW['env'] = ''
    std_UW['pos'] = 'SD:'

    std_US['subject'] = ''
    std_US['target'] = ''
    std_US['env'] = ''
    std_US['pos'] = 'SD:'
    std_WS['subject'] = ''
    std_WS['target'] = ''
    std_WS['env'] = ''
    std_WS['pos'] = 'SD:'

    df_output = df_output.append(WW, ignore_index=True)
    df_output = df_output.append(std_WW, ignore_index=True)
    df_output = df_output.append(UW, ignore_index=True)
    df_output = df_output.append(std_UW, ignore_index=True)
    df_output = df_output.append(US, ignore_index=True)
    df_output = df_output.append(std_US, ignore_index=True)
    df_output = df_output.append(WS, ignore_index=True)
    df_output = df_output.append(std_WS, ignore_index=True)
    df_output.to_csv(path_or_buf=(DATA_ROOT / 'hololens_analysis' / 'summary_1st_analysis_proceed.csv'), index=False)


def summary_holo_data():
    df_holo = pd.read_csv(DATA_ROOT / 'hololens_analysis' / '1st_analysis.csv')
    output = []
    df_output = pd.DataFrame(columns=df_holo.columns)
    for subject in subjects:
        df_subject = df_holo[df_holo['subject'] == subject]
        df_WW = df_subject[(df_subject['env'] == 'W') & (df_subject['pos'] == 'W')]
        df_UW = df_subject[(df_subject['env'] == 'U') & (df_subject['pos'] == 'W')]
        df_US = df_subject[(df_subject['env'] == 'U') & (df_subject['pos'] == 'S')]
        df_WS = df_subject[(df_subject['env'] == 'W') & (df_subject['pos'] == 'S')]
        # print(df_subject.shape[0], df_WW.shape[0], df_UW.shape[0], df_US.shape[0], df_WS.shape[0])

        WW = df_WW.mean()
        WW['env'] = 'W'
        WW['pos'] = 'W'
        UW = df_UW.mean()
        UW['env'] = 'U'
        UW['pos'] = 'W'
        US = df_US.mean()
        US['env'] = 'U'
        US['pos'] = 'S'
        WS = df_WS.mean()
        WS['env'] = 'W'
        WS['pos'] = 'S'
        df_output = df_output.append(WW, ignore_index=True)
        df_output = df_output.append(UW, ignore_index=True)
        df_output = df_output.append(US, ignore_index=True)
        df_output = df_output.append(WS, ignore_index=True)

        # df_output.append(df_WW.mean(),ignore_index=True)
        # df_WW = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'W')]
        # df_UW = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'W')]
        # df_US = df_holo[(df_holo['env'] == 'U') & (df_holo['pos'] == 'S')]
        # df_WS = df_holo[(df_holo['env'] == 'W') & (df_holo['pos'] == 'S')]
        # print(df_holo.shape[0], df_WW.shape[0], df_UW.shape[0], df_US.shape[0], df_WS.shape[0])

    df_output.to_csv(path_or_buf=(DATA_ROOT / 'hololens_analysis' / 'summary_1st_analysis.csv'), index=False)
    print('saved')


def analyse_holo_data():
    outputList = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                print(subject, current_info)
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_holo = file_as_pandas(hololens_data)

                check_holo_file(df_holo, current_info)

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
                for row in df_holo.itertuples(index=False):
                    if row[13] == target_name and previous_target != target_name:
                        list_target_in.append(row[0])
                    if row[13] != target_name and previous_target == target_name:
                        list_target_out.append(row[0])
                    previous_target = row[13]

                num_over = len(list_target_in)
                num_off = len(list_target_out)
                if num_over < 1:
                    print('zero over count')
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
                df_trial = df_holo[df_holo['Timestamp'] > list_target_in[0]]
                dur_over = np.sum(list_duration_in)
                mean_over = np.sum(list_duration_in) / num_over
                sd_over = np.std(list_duration_in, dtype=np.float64)
                angular_diff_mean = df_trial['TargetAngularDistance'].values.mean()
                angular_diff_sd = df_trial['TargetAngularDistance'].values.std()
                angular_max = df_trial['TargetAngularDistance'].values.max()
                angular_min = df_trial['TargetAngularDistance'].values.min()
                head_rotation_x_max = df_trial['HeadRotationX'].values.max()
                head_rotation_y_max = df_trial['HeadRotationY'].values.max()
                head_rotation_z_max = df_trial['HeadRotationZ'].values.max()
                head_rotation_x_min = df_trial['HeadRotationX'].values.min()
                head_rotation_y_min = df_trial['HeadRotationY'].values.min()
                head_rotation_z_min = df_trial['HeadRotationZ'].values.min()
                head_rotation_x_range = head_rotation_x_max - head_rotation_x_min
                head_rotation_y_range = head_rotation_y_max - head_rotation_y_min
                head_rotation_z_range = head_rotation_z_max - head_rotation_z_min
                outputList.append(
                    [subject, *current_info, total_time, frame_rate, start_time, num_over, dur_over, mean_over, sd_over,
                     angular_diff_mean, angular_diff_sd, angular_max, angular_min, head_rotation_x_max,
                     head_rotation_y_max, head_rotation_z_max, head_rotation_x_min, head_rotation_y_min,
                     head_rotation_z_min, head_rotation_x_range, head_rotation_y_range, head_rotation_z_range])
                print('-' * 50)


            except ValueError as err:
                print(subject, current_info, err)
    output = pd.DataFrame(outputList,
                          columns=['subject', 'target', 'env', 'pos', 'block', 'total_time', 'frame_rate', 'start_time',
                                   'num_over', 'dur_over', 'mean_over', 'sd_over', 'angular_diff_mean',
                                   'angular_diff_sd', 'angular_max', 'angular_min', 'head_rotation_x_max',
                                   'head_rotation_y_max', 'head_rotation_z_max', 'head_rotation_x_min',
                                   'head_rotation_y_min',
                                   'head_rotation_z_min', 'head_rotation_x_range', 'head_rotation_y_range',
                                   'head_rotation_z_range'])
    output.to_csv(path_or_buf=(DATA_ROOT / 'hololens_analysis' / '1st_analysis.csv'), index=False)


def check_files():
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                imu_data = get_file_by_info(imu_file_list, current_info)
                eye_data = get_file_by_info(eye_file_list, current_info)
                hololens_data = get_file_by_info(hololens_file_list, current_info)
                df_imu = manipulate_imu(file_as_pandas(imu_data))
                # df_eye = manipulate_eye(file_as_pandas(eye_data))
                df_eye = file_as_pandas(eye_data, refined=True)
                df_holo = file_as_pandas(hololens_data)
                check_file(df_imu, df_eye, df_holo, current_info)

            except ValueError as err:
                print(subject, current_info, err)


def summary_eye_data():
    # ['circle_3d', 'confidence', 'timestamp', 'diameter_3d', 'ellipse',
    #        'location', 'diameter', 'sphere', 'projected_sphere',
    #        'model_confidence', 'model_id', 'model_birth_timestamp', 'theta', 'phi',
    #        'norm_pos', 'topic', 'id', 'method']
    output_list = []
    for subject in subjects:
        [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=True)
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            current_info = [target, env, pos, block]
            try:
                eye_data = get_file_by_info(eye_file_list, current_info)
                df_eye = file_as_pandas(eye_data, refined=True)
                check_refined_eye_file(df_eye, current_info)
                _start_time = df_eye.head(1)['timestamp'].values[0]
                df_trial = df_eye[df_eye['timestamp'] > (_start_time + 1.5)]

                mean_confidence = df_trial['confidence'].mean()
                mean_theta = df_trial['theta'].mean()
                max_theta = df_trial['theta'].max()
                min_theta = df_trial['theta'].min()
                range_theta = max_theta - min_theta
                mean_phi = df_trial['phi'].mean()
                max_phi = df_trial['phi'].max()
                min_phi = df_trial['phi'].min()
                range_phi = max_phi - min_phi
                df_norm_pos = df_trial['norm_pos']
                norm_pos_x = []
                norm_pos_y = []
                for row in df_norm_pos:
                    output = row.split('[')[1]
                    output = output.split(']')[0]
                    output = output.split(',')
                    row_x = output[0]
                    row_y = output[1]
                    if 'Decimal' in output[0]:
                        row_x = output[0].split('(')[1]
                        row_x = row_x.split(')')[0]
                        row_x = ''.join(c for c in row_x if c.isdigit() or c == '.')

                    if 'Decimal' in output[1]:
                        row_y = output[1].split('(')[1]
                        row_y = row_y.split(')')[0]
                        row_y = ''.join(c for c in row_y if c.isdigit() or c == '.')
                    norm_pos_x.append(float(row_x))
                    norm_pos_y.append(float(row_y))
                mean_norm_pos_x = statistics.mean(norm_pos_x)
                max_norm_pos_x = max(norm_pos_x)
                min_norm_pos_x = min(norm_pos_x)
                range_norm_pos_x = max_norm_pos_x - min_norm_pos_x
                mean_norm_pos_y = statistics.mean(norm_pos_y)
                max_norm_pos_y = max(norm_pos_y)
                min_norm_pos_y = min(norm_pos_y)
                range_norm_pos_y = max_norm_pos_y - min_norm_pos_y

                output_list.append(
                    [subject, *current_info, mean_confidence, mean_norm_pos_x, max_norm_pos_x, min_norm_pos_x,
                     range_norm_pos_x, mean_norm_pos_y, max_norm_pos_y, min_norm_pos_y, range_norm_pos_y, mean_phi,
                     max_phi, min_phi, range_phi, mean_theta, max_theta, min_theta,
                     range_theta])
            except ValueError as err:
                print(subject, current_info, err)
    df_output = pd.DataFrame(output_list, columns=[
        'subject', 'target', 'env', 'pos', 'block', 'mean_confidence', 'mean_norm_pos_x', 'max_norm_pos_x',
        'min_norm_pos_x',
        'range_norm_pos_x', 'mean_norm_pos_y', 'max_norm_pos_y', 'min_norm_pos_y', 'range_norm_pos_y',
        'mean_phi', 'max_phi', 'min_phi', 'range_phi', 'mean_theta', 'max_theta', 'min_theta',
        'range_theta'
    ])
    df_output.to_csv(path_or_buf=(DATA_ROOT / 'refined_eye_data_analysis' / '1st_analysis.csv'), index=False)


def get_final_dataset(_target, _env, _pos, _block, _subject):
    [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(_subject, refined=True)
    current_info = [_target, _env, _pos, _block]
    print('-' * 15 + str(current_info) + " / " + str(_subject) + '-' * 15)
    try:
        df_imu, df_eye, df_holo = bring_dataframe(imu_file_list, eye_file_list, hololens_file_list, current_info)
        timestamps = {'imu': df_imu['IMUtimestamp'], 'eye': df_eye['timestamp'], 'holo': df_holo['Timestamp']}
        df_eye_conf = df_eye[df_eye['check'] != 'drop']
        df_eye_low_conf = df_eye[df_eye['check'] == 'drop']
        if round((df_eye_low_conf.shape[0] / df_eye.shape[0]) * 100, 2) > 30:
            raise ValueError('too many low confidence eye data')

        target_name = 'target_' + str(_target)
        previous_target = ''
        list_target_in = []
        for row in df_holo.itertuples(index=False):
            if row[13] == target_name and previous_target != target_name:
                list_target_in.append(row[0])
        start_time = list_target_in[0]

        timestamps['eye_conf'] = df_eye_conf['timestamp']
        df_eye_phi = pd.Series(list(map(float, df_eye_conf['phi'])))
        df_eye_the = pd.Series(list(map(float, df_eye_conf['theta'])))
        df_eye_x = pd.Series(list(map(float, df_eye_conf['norm_x'])))
        df_eye_y = pd.Series(list(map(float, df_eye_conf['norm_y'])))
        low_conf_start = []
        low_conf_end = []
        for index, row in df_eye.iterrows():
            if index == 0:
                if df_eye.loc[index, 'check'] == 'drop':
                    low_conf_start.append(df_eye.loc[index, 'timestamp'])
            else:
                if df_eye.loc[index, 'check'] == 'drop' and df_eye.loc[index - 1, 'check'] != 'drop':
                    low_conf_start.append(df_eye.loc[index, 'timestamp'])
                if (df_eye.loc[index, 'check'] != 'drop') and (df_eye.loc[index - 1, 'check'] == 'drop'):
                    low_conf_end.append(df_eye.loc[index, 'timestamp'])

        # print('start', low_conf_start)
        # print('end', low_conf_end)
        # filtered_phi, filtered_the = eye_one_euro_filtering(df_eye_phi, df_eye_the)
        # filtered_x, filtered_y = eye_one_euro_filtering(df_eye_x, df_eye_y)
        filtered_phi = df_eye_phi.rolling(window=2).mean()
        filtered_the = df_eye_the.rolling(window=2).mean()
        filtered_x = df_eye_x.rolling(window=2).mean()
        filtered_y = df_eye_y.rolling(window=2).mean()
        rs = []
        thetas = []
        phis = []
        for index, row in df_holo.iterrows():
            x = row['TargetPositionX'] - row['HeadPositionX']
            y = row['TargetPositionY'] - row['HeadPositionY']
            z = row['TargetPositionZ'] - row['HeadPositionZ']
            [r, theta, phi] = asSpherical([x, z, y])
            rs.append(r)
            thetas.append(90 - theta)
            phis.append(90 - phi)

        df_holo['R'] = rs
        df_holo['Theta'] = thetas
        df_holo['Phi'] = phis

        [interpolation_imu_z, interpolation_imu_y, interpolation_imu_x] = get_interpolation_function(
            timestamps['imu'], df_imu['rotationZ'], df_imu['rotationY'], df_imu['rotationX']
        )
        [interpolation_eye_phi, interpolation_eye_the, interpolation_eye_x,
         interpolation_eye_y] = get_interpolation_function(
            timestamps['eye_conf'], df_eye_phi, df_eye_the, df_eye_x, df_eye_y
        )
        [interpolation_holo_x, interpolation_holo_y, interpolation_holo_z,
         interpolation_holo_theta, interpolation_holo_phi] = get_interpolation_function(
            timestamps['holo'], df_holo['HeadRotationX'], df_holo['HeadRotationY'], df_holo['HeadRotationZ'],
            df_holo['Theta'], df_holo['Phi']
        )
        interpolation_eye_phi_raw = interpolate.interp1d(timestamps['eye_conf'], df_eye_phi)
        interpolation_eye_the_raw = interpolate.interp1d(timestamps['eye_conf'], df_eye_the)
        interpolation_eye_x_raw = interpolate.interp1d(timestamps['eye_conf'], df_eye_x)
        interpolation_eye_y_raw = interpolate.interp1d(timestamps['eye_conf'], df_eye_y)

        x_imu = make_x_axis(timestamps['imu'], interval=0.005)
        x_eye = make_x_axis(timestamps['eye'], interval=0.005)

        x_holo = make_x_axis(timestamps['holo'], interval=0.005)
        x_test = np.arange(1, 6, 0.005)
        x_test_eye = x_test
        x_test_imu = x_test

        x_axis = {'imu': x_imu, 'eye': x_eye, 'holo': x_holo, 'test': x_test}
        dataframe = {'imu': df_imu, 'eye': df_eye, 'holo': df_holo, 'eye_conf': df_eye_conf,
                     'eye_low_conf': df_eye_low_conf}
        interpolations = {
            'imuX': interpolation_imu_x, 'imuY': interpolation_imu_y, 'imuZ': interpolation_imu_z,
            'holoX': interpolation_holo_x, 'holoY': interpolation_holo_y, 'holoZ': interpolation_holo_z,
            'holoPhi': interpolation_holo_phi, 'holoThe': interpolation_holo_theta,
            'eyeX_raw': interpolation_eye_x_raw, 'eyeY_raw': interpolation_eye_y_raw,
            'eyeX': interpolation_eye_x, 'eyeY': interpolation_eye_y,
            'eyePhi_raw': interpolation_eye_phi_raw, 'eyeThe_raw': interpolation_eye_the_raw,
            'eyePhi': interpolation_eye_phi, 'eyeThe': interpolation_eye_the,
            # 'targetVertical': interpolation_holo_x + interpolation_holo_theta,
            # 'targetHorizontal': interpolation_holo_y - interpolation_holo_phi
        }
        # Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
        # Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi

        low_conf = {'start': np.asarray(low_conf_start), 'end': np.asarray(low_conf_end)}
        return x_axis, dataframe, low_conf, list_target_in, interpolations, current_info
    except (TypeError, ValueError, KeyError) as err:
        print(_subject, current_info, err)


def save_dataframe_as_one():
    data = pd.DataFrame(
        columns=['subject', 'target', 'env', 'pos', 'block', 'x_axis', 'dataframe', 'low_conf', 'list_target_in',
                 'interpolations'])
    for target, env, pos, block, subject in itertools.product(targets, envs, poss, blocks, subjects):
        try:
            x_axis, dataframe, low_conf, list_target_in, interpolations, current_info = get_final_dataset(
                target, env, pos, block, subject)
            data = data.append(
                {'subject': subject, 'target': target, 'env': env, 'pos': pos, 'block': block, 'x_axis': x_axis,
                 'dataframe': dataframe, 'low_conf': low_conf, 'list_target_in': list_target_in,
                 'interpolations': interpolations}, ignore_index=True)
            print(target, env, pos, block, subject, 'appended')
        except (TypeError, ValueError) as err:
            print(err)
            data = data.append({'subject': subject, 'target': target, 'env': env, 'pos': pos, 'block': block},
                               ignore_index=True)
    # %%
    data.to_pickle("whole.pkl")


def target_positional_analysis():
    for subject in subjects:
        # [imu_file_list, eye_file_list, hololens_file_list] = get_one_subject_files(subject, refined=
        sns.set()
        fig, axs = plt.subplots(2, 1, figsize=[6, 12], sharex=True)
        whole_eye_vertical = np.array([])
        whole_eye_horizontal = np.array([])
        whole_dff_vertical = np.array([])
        whole_dff_horizontal = np.array([])
        whole_target_vertical = np.array([])
        whole_target_horizontal = np.array([])
        whole_dff_norm_vertical = np.array([])
        whole_dff_norm_horizontal = np.array([])

        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            try:
                x_axis, dataframe, low_conf, list_target_in, interpolations, current_info = get_final_dataset(
                    target, env, pos, block, subject)

                #
                # axs[0][0].set_title(str(current_info) + 'vertical')
                # axs[1][0].set_title('horizontal')
                # eye - analysis
                timeline = parse_timeline(low_conf['start'], low_conf['end'])
                target_verticals = np.array([])
                target_horizontals = np.array([])
                eye_linregress_verticals = np.array([])
                eye_linregress_horizontals = np.array([])
                eye_filtered_verticals = np.array([])
                eye_filtered_horizontals = np.array([])
                times = np.array([])

                for time in timeline:
                    t1 = np.where(((x_axis['test'] > time[0]) & (x_axis['test'] < time[1])))
                    t = np.arange(math.ceil(time[0] * 100) / 100, math.floor(time[1] * 100) / 100, 0.005)

                    if len(t1[0]) < 50:
                        # print('short timeline... drop this')
                        continue
                    times = np.append(times, t)

                    target_vertical = interpolations['holoX'](t) + interpolations['holoThe'](t)
                    target_horizontal = interpolations['holoY'](t) - interpolations['holoPhi'](t)
                    eye_vertical = (-1 * interpolations['eyeY_raw'](t))
                    eye_horizontal = (-1 * interpolations['eyeX_raw'](t))
                    filtered_eye_vertical = one_euro(eye_vertical)
                    filtered_eye_horizontal = one_euro(eye_horizontal)
                    filtered_target_vertical = one_euro(target_vertical)
                    filtered_target_horizontal = one_euro(target_horizontal)

                    slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
                        filtered_eye_vertical, filtered_target_vertical)
                    slope_horizontal, intercept_horizontal, r_horizontal, p_horizontal, std_err_horizontal = stats.linregress(
                        filtered_eye_horizontal, filtered_target_horizontal)

                    if p_vertical > 0.05 or p_horizontal > 0.05:
                        continue
                    eye_linregress_vertical = slope_vertical * filtered_eye_vertical + intercept_vertical
                    eye_linregress_horizontal = slope_horizontal * filtered_eye_horizontal + intercept_horizontal
                    eye_normalised_vertical = normalize(eye_vertical, target_vertical)

                    target_verticals = np.append(target_verticals, filtered_target_vertical)
                    target_horizontals = np.append(target_horizontals, filtered_target_horizontal)

                    eye_linregress_verticals = np.append(eye_linregress_verticals, eye_linregress_vertical)
                    eye_linregress_horizontals = np.append(eye_linregress_horizontals, eye_linregress_horizontal)
                    eye_filtered_verticals = np.append(eye_filtered_verticals, filtered_eye_vertical)
                    eye_filtered_horizontals = np.append(eye_filtered_horizontals, filtered_eye_horizontal)
                    # axs[0][0].plot(t,filtered_target_vertical,color='black')
                    # print(r_vertical,p_vertical)
                    # axs[0][0].plot(t,eye_raw_vertical,color='blue')
                    #
                    # axs[1][0].plot(t, filtered_target_horizontal, color='black')
                    # axs[1][0].plot(t, eye_raw_horizontal, color='blue')
                    #
                    # axs[0][1].scatter(eye_vertical, target_vertical, marker='x', alpha=0.5)
                    # axs[0][1].plot(filtered_eye_vertical, eye_raw_vertical)
                    #
                    # axs[1][1].scatter(eye_horizontal, target_horizontal, marker='x', alpha=0.5)
                    # axs[1][1].plot(filtered_eye_horizontal, eye_raw_horizontal)

                # axs[1][0].vlines(list_target_in, ymax=0.1, ymin=-0.1)
                # axs[0][0].vlines(list_target_in, ymax=0.1, ymin=-0.1)
                if len(eye_linregress_verticals) < 5:
                    print('no useful data')
                    continue
                slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
                    eye_filtered_verticals, target_verticals)
                slope_horizontal, intercept_horizontal, r_horizontal, p_horizontal, std_err_horizontal = stats.linregress(
                    eye_filtered_horizontals, target_horizontals)

                whole_dff_vertical = np.append(whole_dff_vertical, target_verticals - eye_linregress_verticals)
                whole_dff_horizontal = np.append(whole_dff_horizontal, target_horizontals - eye_linregress_horizontals)
                whole_target_vertical = np.append(whole_target_vertical, target_verticals)
                whole_target_horizontal = np.append(whole_target_horizontal, target_horizontals)
                whole_dff_norm_vertical = np.append(whole_dff_norm_vertical, eye_normalised_vertical)
                whole_dff_norm_horizontal = np.append(whole_dff_norm_horizontal, eye_normalised_vertical)
                whole_eye_vertical = np.append(whole_eye_vertical, eye_filtered_verticals)
                whole_eye_horizontal = np.append(whole_eye_horizontal, eye_filtered_horizontals)

                # sns.regplot(eye_filtered_verticals,target_verticals,ax=axs[2][0],marker='+',scatter_kws={'color':'k','alpha':0.3})
                # sns.regplot(eye_filtered_horizontals,target_horizontals,ax=axs[2][1],marker='+',scatter_kws={'color':'k','alpha':0.3})

            except (TypeError, ValueError) as err:
                print(err)
                continue
        sns.scatterplot(whole_target_horizontal, whole_target_vertical, ax=axs[0], markers='+', alpha=0.3)
        confidence_ellipse(whole_target_horizontal, whole_target_vertical, n_std=3, edgecolor='firebrick', ax=axs[0])
        sns.scatterplot(whole_dff_horizontal, whole_dff_vertical, ax=axs[1], markers='+', alpha=0.3)
        confidence_ellipse(whole_dff_horizontal, whole_dff_vertical, n_std=3, edgecolor='fuchsia', ax=axs[1])
        confidence_ellipse(whole_target_horizontal, whole_target_vertical, n_std=3, edgecolor='firebrick', ax=axs[1])

        axs[0].axis('equal')
        axs[1].axis('equal')
        axs[0].set_title('target')
        axs[1].set_title('compensated-linregress')

        # sns.distplot(whole_dff_vertical, fit=stats.norm, kde=True, label='compenstaed-linregress', ax=axs[0][0])
        # axs[0][0].set_title("linregress-vertical")
        # sns.distplot(whole_target_vertical, fit=stats.norm, kde=True, label='target', ax=axs[0][1])
        # axs[0][1].set_title('target-vertical')
        # sns.distplot(whole_dff_horizontal,fit=stats.norm,kde=True,label= 'compensated-linregress',ax=axs[1][0])
        # axs[1][0].set_title('linregress-horizontal')
        # sns.distplot(whole_target_horizontal,fit=stats.norm,kde=True,ax=axs[1][1])
        # axs[1][1].set_title('target-horizontal')
        # axs[2][0].set_title('vertical')
        # axs[2][1].set_title('horizontal')
        plt.legend()
        fig.suptitle('' + str(subject), horizontalalignment='left', verticalalignment='top', fontsize=15)
        plt.savefig('target_plots_' + str(subject - 200) + '.pdf')
        # fig.tight_layout()
        plt.show()
        print('drawing plot', current_info, subject)


def bring_dataframe(imu_file_list, eye_file_list, hololens_file_list, current_info):
    # try:
    df_imu = manipulate_imu(file_as_pandas(get_file_by_info(imu_file_list, current_info)))
    df_eye = file_as_pandas(get_file_by_info(eye_file_list, current_info), refined=True)
    df_holo = file_as_pandas(get_file_by_info(hololens_file_list, current_info))
    check_imu_file(df_imu, current_info)
    check_refined_eye_file(df_eye, current_info)
    check_holo_file(df_holo, current_info)
    df_imu['IMUtimestamp'] = df_imu['IMUtimestamp'] - df_imu['IMUtimestamp'].head(1).values[0] - 0.05
    df_eye['timestamp'] = df_eye['timestamp'] - df_eye['timestamp'].head(1).values[0] - 0.05
    df_holo['Timestamp'] = df_holo['Timestamp'] - df_holo['Timestamp'].head(1).values[0]
    # circle_3d	confidence	timestamp	diameter_3d	ellipse	location	diameter	sphere	projected_sphere	model_confidence	model_id	model_birth_timestamp	theta	phi	norm_pos	topic	id	method	norm_x	norm_y
    df_eye["check"] = np.nan
    for index, row in df_eye.iterrows():
        if row['confidence'] < 0.6 and index != 0:
            df_eye.loc[index - 2, 'check'] = 'drop'
            df_eye.loc[index - 1, 'check'] = 'drop'
            df_eye.loc[index, 'check'] = 'drop'
            df_eye.loc[index + 1, 'check'] = 'drop'
            df_eye.loc[index + 2, 'check'] = 'drop'

    # except ValueError as err:
    #     print(current_info, err)
    return df_imu, df_eye, df_holo

# @logging_time
def slopes(data):
    vertical_slope=np.array([])
    horizontal_slope = np.array([])
    for index, row in data.iterrows():
        # print(row['x_axis'])
        if type(row['x_axis']) is not dict:
            # print('empty row')
            continue

        # each trial
        timeline = parse_timeline(row['low_conf']['start'], row['low_conf']['end'])
        for time in timeline:
            t1 = np.where(((row['x_axis']['test'] > time[0]) & (row['x_axis']['test'] < time[1])))
            if len(t1[0]) < 50:
                # print('too short timeline')
                continue
            t = np.arange(math.ceil(time[0] * 100) / 100, math.floor(time[1] * 100) / 100, 0.005)
            target_vertical = row['interpolations']['holoX'](t) + row['interpolations']['holoThe'](t)
            target_horizontal = row['interpolations']['holoY'](t) - row['interpolations']['holoPhi'](t)
            eye_vertical = (-1 * row['interpolations']['eyeY_raw'](t))
            eye_horizontal = (-1 * row['interpolations']['eyeX_raw'](t))
            filtered_eye_vertical = one_euro(eye_vertical)
            filtered_eye_horizontal = one_euro(eye_horizontal)
            filtered_target_vertical = one_euro(target_vertical)
            filtered_target_horizontal = one_euro(target_horizontal)

            slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
                filtered_eye_vertical, filtered_target_vertical)
            slope_horizontal, intercept_horizontal, r_horizontal, p_horizontal, std_err_horizontal = stats.linregress(
                filtered_eye_horizontal, filtered_target_horizontal)
            if p_vertical > 0.05 or p_horizontal > 0.05:
                continue
            vertical_slope = np.append(vertical_slope,slope_vertical)
            horizontal_slope = np.append(horizontal_slope,slope_horizontal)

    return vertical_slope, horizontal_slope

@logging_time
def from_pickle(_filename):
    data = pd.read_pickle(_filename)
    subject_data = lambda x: data[data['subject'] == x]
    subject = {x: subject_data(x) for x in subjects}

    vertical_slope, horizontal_slope = slopes(subject[201])
    # print(vertical_slope,horizontal_slope)
    # plt.hist(vertical_slope)
    # sns.distplot(vertical_slope)
    # plt.figure(1)
    # plt.hist(horizontal_slope)
    # sns.distplot(horizontal_slope)
    print('horizontal slope',horizontal_slope.mean())
    print('vertical_slope',vertical_slope.mean())
    return data, horizontal_slope.mean(), vertical_slope.mean()
    # for index, row in data.iterrows():
    #     # print(row['x_axis'])
    #     if type(row['x_axis']) is not dict:
    #         # print('empty row')
    #         continue
    #
    #     # each trial
    #     timeline = parse_timeline(row['low_conf']['start'], row['low_conf']['end'])
    #     for time in timeline:
    #         t1 = np.where(((row['x_axis']['test'] > time[0]) & (row['x_axis']['test'] < time[1])))
    #         if len(t1[0]) < 50:
    #             # print('too short timeline')
    #             continue
    #         t = np.arange(math.ceil(time[0] * 100) / 100, math.floor(time[1] * 100) / 100, 0.005)
    #         target_vertical = row['interpolations']['holoX'](t) + row['interpolations']['holoThe'](t)
    #         target_horizontal = row['interpolations']['holoY'](t) - row['interpolations']['holoPhi'](t)
    #         eye_vertical = (-1 * row['interpolations']['eyeY_raw'](t))
    #         eye_horizontal = (-1 * row['interpolations']['eyeX_raw'](t))
    #         filtered_eye_vertical = one_euro(eye_vertical)
    #         filtered_eye_horizontal = one_euro(eye_horizontal)
    #         filtered_target_vertical = one_euro(target_vertical)
    #         filtered_target_horizontal = one_euro(target_horizontal)
    #         normalized_eye_vertical = (filtered_eye_vertical - filtered_eye_vertical.mean())*vertical_slope.mean() + filtered_target_vertical.mean()
    #         sns.lineplot(t,filtered_target_vertical)
    #         sns.lineplot(t,normalized_eye_vertical)
    #         plt.show()
    # plt.show()


    # print(sub_201)


def main():
    pass


if __name__ == "__main__":
    create_eye_csv()
    # from_pickle('whole.pkl')
    # data = pd.read_pickle('whole.pkl')
    # int1 = data.head(1) ['interpolations'][0]
    # print(int1['imuX']([2,3,4]))
