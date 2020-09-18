"""
NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi

holo.columns
Index(['timestamp', 'head_position', 'head_rotation', 'head_forward',
       'target_position', 'target_entered', 'angular_distance',
       'head_position_x', 'head_position_y', 'head_position_z',
       'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
       'head_forward_x', 'head_forward_y', 'head_forward_z',
       'target_position_x', 'target_position_y', 'target_position_z'],
      dtype='object')

imu.columns
Index(['IMUtimestamp', 'rotationX', 'rotationY', 'rotationZ'], dtype='object')

eye.columns
Index(['circle_3d', 'confidence', 'timestamp', 'diameter_3d', 'ellipse',
       'location', 'diameter', 'sphere', 'projected_sphere',
       'model_confidence', 'model_id', 'model_birth_timestamp', 'theta', 'phi',
       'norm_pos', 'topic', 'id', 'method', 'python_timestamp', 'norm_x',
       'norm_y'],
      dtype='object')
"""
# %% Importing
from plotly.subplots import make_subplots
from analysing_functions import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = 'browser'


# %% test one trial
def summary_one_trial(target, env, block, subject):
    # 0, "U", 4, 316
    holo, imu, eye = bring_data(target, env, block, subject)
    shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
    imu.IMUtimestamp = imu.IMUtimestamp - shift_time
    eye.timestamp = eye.timestamp - shift_time
    # eye.norm_x = eye.norm_x - eye.norm_x.mean()
    # eye.norm_y = eye.norm_y - eye.norm_y.mean()

    # imu = imu_to_vector(imu)
    new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
    new_imu = imu_to_vector(new_imu)
    new_holo = holo_to_vector(new_holo)
    # new_holo = new_holo[new_holo.timestamp >= 1.5]
    # new_imu = new_imu[new_imu.timestamp >= 1.5]
    # new_eye = new_eye[new_eye.timestamp >= 1.5]
    rot, rmsd = R.align_vectors(new_holo[['head_forward_x', 'head_forward_y', 'head_forward_z']],
                                new_imu[['vector_x', 'vector_y', 'vector_z']])
    applied_imu = rot.apply(new_imu[['vector_x', 'vector_y', 'vector_z']])

    fs = 120
    fc = 4
    w = fc / (fs / 2)
    mincutoff = 3.0
    beta = 0.01
    b, a = signal.butter(3, w, 'low', analog=False)
    # filtered_norm_x= pd.Series(signal.filtfilt(b, a, new_eye.norm_x))
    filtered_norm_x = pd.Series(one_euro(new_eye.norm_x, beta=beta, mincutoff=mincutoff))
    filtered_norm_x.index = new_eye.index
    new_eye['filtered_norm_x'] = filtered_norm_x
    filtered_norm_y = pd.Series(one_euro(new_eye.norm_y, beta=beta, mincutoff=mincutoff))
    filtered_norm_y.index = new_eye.index
    new_eye['filtered_norm_y'] = filtered_norm_y
    filtered_target_horizontal = pd.Series(one_euro(new_holo.TargetHorizontal, beta=beta, mincutoff=mincutoff))
    filtered_target_horizontal.index = new_holo.index
    new_holo['filtered_TargetHorizontal'] = filtered_target_horizontal
    filtered_target_vertical = pd.Series(one_euro(new_holo.TargetVertical, beta=beta, mincutoff=mincutoff))
    filtered_target_vertical.index = new_holo.index
    new_holo['filtered_TargetVertical'] = filtered_target_vertical

    # get multiple?
    # multiple_horizontal = interpolated_target_horizontal.abs().mean() / eye.norm_x.abs().mean()
    # multiple_vertical = interpolated_target_vertical.abs().mean() / eye.norm_y.abs().mean()
    offset = int(120 * 1.5)
    multiple_horizontal = new_holo.filtered_TargetHorizontal[offset:].abs().mean() / new_eye.filtered_norm_x[
                                                                                     offset:].abs().mean()
    multiple_vertical = new_holo.filtered_TargetVertical[offset:].abs().mean() / new_eye.filtered_norm_y[
                                                                                 offset:].abs().mean()

    # HORIZONTAL
    fig_horizontal = make_subplots(rows=4, cols=1)
    fig_horizontal.update_layout(title=dict(text=str(multiple_horizontal), font={'size': 30}))
    # EYE
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp, y=(new_eye.norm_x * multiple_horizontal), name='eye-x', opacity=0.5), row=1,
        col=1)
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp, y=(new_eye.filtered_norm_x * multiple_horizontal), name='filtered-eye-x'),
        row=1, col=1)
    # HOLO
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetHorizontal, name='filtered_target-x'), row=2, col=1)
    # slope_horizontal, intercept_horizontal = normalize(new_imu.rotationZ, new_holo.head_rotation_y)
    # fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_imu.rotationZ * slope_horizontal + intercept_horizontal, name='imu-x'), row=2, col=1)
    fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_y - holo.Phi, name='target-x'), row=2,
                             col=1)
    # TARGET
    # fig_horizontal.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetHorizontal, name='target-x'), row=3, col=1)
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetHorizontal, name='filtered-target-x'), row=3, col=1)
    fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.TargetHorizontal, name='target-x', opacity=0.3),
                             row=3, col=1)
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp, y=-new_eye.filtered_norm_x * multiple_horizontal, name='filtered-eye-x'),
        row=3,
        col=1)
    # COMPENSATED
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp,
                   y=new_holo.filtered_TargetHorizontal + new_eye.filtered_norm_x * multiple_horizontal,
                   name='compensated'), row=4,
        col=1)

    # VERTICAL
    fig_vertical = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=True,
                                 subplot_titles=("eye", 'head', 'target', 'compensation'))
    fig_vertical.update_layout(title=dict(text=str(multiple_vertical), font=dict(size=30)))
    # EYE
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=(new_eye.norm_y * multiple_vertical), name='eye-y', opacity=0.5), row=1,
        col=1)
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=(new_eye.filtered_norm_y * multiple_vertical), name='filtered-eye-y'), row=1,
        col=1)
    # HOLO
    fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_holo.head_rotation_x, name='head-y'), row=2, col=1)
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetVertical, name='filtered-target-y'), row=2, col=1)
    # fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp, y=new_imu.rotationX, name='imu-y'), row=2, col=1)
    fig_vertical.add_trace(
        go.Scatter(x=holo.timestamp, y=holo.Theta + holo.head_rotation_x, name='target-y', opacity=0.3), row=2, col=1)
    # TARGET
    # fig_vertical.add_trace(go.Scatter(x=holo.timestamp, y=holo.TargetVertical, name='target-y'), row=3, col=1)
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=new_holo.filtered_TargetVertical, name='filtered-target-y'), row=3, col=1)
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=(-new_eye.filtered_norm_y * multiple_vertical), name='eye-y'), row=3, col=1)
    # fill='tonexty
    # COMPENSATED
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp, y=new_holo.TargetVertical + new_eye.norm_y * multiple_vertical,
                   name='unfiltered-compensate', opacity=0.3),
        row=4, col=1)
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp,
                   y=new_holo.filtered_TargetVertical + new_eye.filtered_norm_y * multiple_vertical,
                   name='compensated-y'),
        row=4,
        col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=1, col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=2, col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=3, col=1)
    fig_vertical.update_yaxes(range=[-2, 2], row=4, col=1)
    window = 120
    estimation_horizontal = []
    estimation_vertical = []
    for index in range(new_eye.shape[0] - window - 1):
        temp_holo = new_holo[index:index + window]
        temp_eye = new_eye[index:index + window]
        slope_horizontal, intercept_horizontal = normalize(temp_eye.filtered_norm_x,
                                                           temp_holo.filtered_TargetHorizontal)
        slope_vertical, intercept_vertical = normalize(temp_eye.filtered_norm_y, temp_holo.filtered_TargetVertical)
        estimated_horizontal = new_eye.iloc[index + window + 1][
                                   'filtered_norm_x'] * slope_horizontal + intercept_horizontal
        estimated_vertical = new_eye.iloc[index + window + 1]['filtered_norm_y'] * slope_vertical + intercept_vertical
        estimation_horizontal.append(estimated_horizontal)
        estimation_vertical.append(estimated_vertical)
    fig_horizontal.add_trace(go.Scatter(x=new_eye.timestamp[window + 1:], y=estimation_horizontal, name='real-time-x'),
                             row=3, col=1)
    fig_horizontal.add_trace(
        go.Scatter(x=new_eye.timestamp[window + 1:],
                   y=new_holo.filtered_TargetHorizontal[window + 1:] - estimation_horizontal, name='estimation-x'),
        row=4, col=1)
    fig_vertical.add_trace(go.Scatter(x=new_eye.timestamp[window + 1:], y=estimation_vertical, name='real-time-y'),
                           row=3, col=1)
    fig_vertical.add_trace(
        go.Scatter(x=new_eye.timestamp[window + 1:],
                   y=new_holo.filtered_TargetVertical[window + 1:] - estimation_vertical, name='estimation-y'),
        row=4, col=1)
    fig_horizontal.show()
    fig_vertical.show()


# %%
# import plotly.figure_factory as ff
#
# hist_data = [new_holo.filtered_TargetHorizontal + new_eye.filtered_norm_x * multiple_horizontal,
#              new_holo.filtered_TargetVertical + new_eye.filtered_norm_y * multiple_vertical,
#              new_holo.TargetHorizontal, new_holo.TargetVertical]
# group_labels = ['horizontal', 'vertical', 'target_horizontal', 'target_vertical']
# fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2, show_hist=False, curve_type='normal')
# fig.show()


# %% check the distribution fo x-pos and angle distance
def check_distribution():
    # from plotly.subplots import make_subplots
    # from analysing_functions import *
    # # IF you are using Pycharm
    # import plotly.io as pio
    # from scipy import fftpack, signal

    pio.renderers.default = 'browser'
    subjects = range(301, 317)
    envs = ["W", "U"]
    targets = range(8)
    blocks = range(1, 5)

    W_startpositionX = []
    W_startAngles = []
    W_walklengths = []
    U_startpositionX = []
    U_startAngles = []
    U_walklengths = []
    for subject in subjects:
        for env, target, block in itertools.product(['W'], targets, blocks):
            try:
                holo = bring_hololens_data(target, env, block, subject)

                W_startpositionX.append(holo.iloc[5]['head_position_x'])
                W_startAngles.append(holo.iloc[5]['angular_distance'])
                walk = holo['head_position_z'].iloc[-1] - holo['head_position_z'].iloc[0]
                W_walklengths.append(walk)
                print(env, target, block, subject)
                pass
            except Exception as e:
                print(e.args)
    for subject in subjects:
        for env, target, block in itertools.product(['U'], targets, blocks):
            try:
                holo = bring_hololens_data(target, env, block, subject)

                U_startpositionX.append(holo.iloc[5]['head_position_x'])
                U_startAngles.append(holo.iloc[5]['angular_distance'])
                walk = holo['head_position_z'].iloc[-1] - holo['head_position_z'].iloc[0]
                U_walklengths.append(walk)
                print(env, target, block, subject)
                pass
            except Exception as e:
                print(e.args)
    from scipy.stats import ks_2samp, wasserstein_distance
    print(ks_2samp(W_startpositionX, W_startAngles))
    print(ks_2samp(W_startpositionX, U_startAngles))
    print(ks_2samp(W_startpositionX, U_startpositionX))
    print(ks_2samp(W_startpositionX, W_startpositionX))
    print(ks_2samp(U_startpositionX, W_startAngles))
    print(ks_2samp(U_startpositionX, U_startAngles))
    print(ks_2samp(U_startpositionX, U_startpositionX))
    print(ks_2samp(U_startpositionX, W_startpositionX))

    # #%%
    # import plotly.figure_factory as ff
    #
    # fig = ff.create_distplot([W_startpositionX,U_startpositionX],['W','U'],bin_size=0.02,show_rug=False)
    # fig.show()
    # W_startAngles = pd.Series(W_startAngles)
    # W_startAngles = W_startAngles[W_startAngles<150]
    # fig = ff.create_distplot([W_startAngles,U_startAngles],['W','U'],bin_size=0.2,show_rug=False)
    # fig.show()


# %% dwell performances
from analysing_functions import *


def angle_velocity(_head_forward, _head_forward2, _time):
    import vg
    if type(_head_forward2) is not dict: return None
    vector1 = np.array([_head_forward['x'], _head_forward['y'], _head_forward['z']])
    vector2 = np.array([_head_forward2['x'], _head_forward2['y'], _head_forward2['z']])
    return vg.angle(vector1, vector2) / _time



# subjects = range(307, 308)
subjects = range(301, 317)
envs = ['U', 'W']
targets = range(8)
blocks = range(1, 5)
# threshold = 0.2
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

dwell_output = []
for subject, env, target, block, threshold in itertools.product(subjects, envs, targets, blocks, thresholds):
    print(target, env, block, subject, threshold)
    holo = bring_hololens_data(target, env, block, subject)
    holo['head_forward_next'] = holo.head_forward.shift(1)
    holo['time_interval'] = holo.timestamp.diff()
    holo['angle_speed'] = holo.apply(lambda x: angle_velocity(x.head_forward, x.head_forward_next, x.time_interval),
                                     axis=1)

    success = pd.DataFrame(holo.target_entered, )
    success['next_frame'] = success.target_entered.shift(1)
    success['start'] = (success.target_entered == True) & (success.next_frame == False)
    success['end'] = (success.target_entered == False) & (success.next_frame == True)

    starts = np.array(success.start[success.start == True].index)
    ends = np.array(success.end[success.end == True].index)
    if starts[-1] > ends[-1]:  starts = np.delete(starts, -1)
    if starts[0] > ends[0]: ends = np.delete(ends, -1)
    durations = ends - starts
    freq = 60

    dwell = np.argwhere(durations > threshold * freq)
    dwell = [x[0] for x in dwell]

    _no_over = True if len(starts) == 0 else False
    if _no_over == True:
        print('no over')
        dwell_output.append(dict(subject=subject,
                                 environment=env,
                                 target=target,
                                 block=block,
                                 threshold=threshold,
                                 no_over=_no_over, ))
        continue;  # check there is an over-target
    if len(dwell) <= 0:
        print('no dwell')
        dwell_output.append(dict(subject=subject,
                                 environment=env,
                                 target=target,
                                 block=block,
                                 threshold=threshold,
                                 dwell_success=False))
        continue;
    # BASIC STATS
    _start_time = holo.timestamp[starts[0]]  # When is the first target-over
    _num_over = len(starts)  # How many times of target-over
    duration_times = [holo.timestamp[ends[i]] - holo.timestamp[starts[i]] for i in range(len(starts))]
    _total_dur_over = sum(duration_times)
    _mean_dur_over = _total_dur_over / _num_over
    _angle_dist_mean = holo.angular_distance.mean()
    _longest_over = max(duration_times)

    # DWELL Stats
    _dwell_success = True if len(dwell) > 0 else False  # Success rate
    _dwell_time = holo.timestamp[starts[dwell[0]]]  # first dwell timestamp

    # SPEED
    _mean_speeds = []
    _vel_ins = []
    _vel_outs = []
    _last100s = []
    for i in range(len(dwell)):
        _mean_speed = holo.iloc[starts[dwell[i]]:ends[dwell[i]]].angle_speed.mean()
        _vel_in = holo.iloc[starts[dwell[i]]].angle_speed
        _vel_out = holo.iloc[ends[dwell[i]]].angle_speed
        _last100 = holo.iloc[ends[dwell[i]] - 6:ends[dwell[i]]].angle_speed.mean()
        _mean_speeds.append(_mean_speed)
        _vel_ins.append(_vel_in)
        _vel_outs.append(_vel_out)
        _last100s.append(_last100)

    _prior_count = int(dwell[0])

    output = dict(
        subject=subject,
        environment=env,
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
    dwell_output.append(output)
summary2 = pd.DataFrame(dwell_output, )

# %% basic summary

summary_sub = dict([])
for subject, env in itertools.product(subjects, envs):
    summary_sub[(subject, env)] = summary2.loc[(summary2['subject'] == subject) & (summary2['environment'] == env)]


# %%
def list_to_mean(data):
    if type(data) is not list:
        return None
    return sum(data) / len(data)

def make_summary_proceed():
    summary_proceed = []
    for subject, env in itertools.product(subjects, envs):
        data = summary_sub[(subject, env)]
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

    df_summary_proceed.to_excel("summary_proceed.xlsx")

#%%
from plotly.subplots import make_subplots
from analysing_functions import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = 'browser'
subjects = range(301, 317)
envs = ["W", "U"]
targets = range(8)
blocks = range(1, 5)

target = 3
env='U'
block = 3
subject=301

holo, imu, eye = bring_data(target, env, block, subject)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
imu.IMUtimestamp = imu.IMUtimestamp - shift_time
eye.timestamp = eye.timestamp - shift_time
# eye.norm_x = eye.norm_x - eye.norm_x.mean()
# eye.norm_y = eye.norm_y - eye.norm_y.mean()

# imu = imu_to_vector(imu)
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
new_imu = imu_to_vector(new_imu)
new_holo = holo_to_vector(new_holo)
# new_holo = new_holo[new_holo.timestamp >= 1.5]
# new_imu = new_imu[new_imu.timestamp >= 1.5]
# new_eye = new_eye[new_eye.timestamp >= 1.5]

fs = 120
fc = 4
w = fc / (fs / 2)
mincutoff = 3.0
beta = 0.01

new_eye['filtered_norm_x'] = one_euro(new_eye.norm_x, beta=beta, mincutoff=mincutoff)
new_eye['filtered_norm_y'] = one_euro(new_eye.norm_y, beta=beta, mincutoff=mincutoff)
# %%
