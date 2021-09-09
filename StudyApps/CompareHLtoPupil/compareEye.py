# %%
import pandas as pd
import demjson
import plotly.graph_objects as go
import plotly.io as pio

import warnings

warnings.simplefilter(action='ignore')

pio.renderers.default = 'browser'

# HL = pd.read_json('FittsLaw40.25813.json')
# noisy
# HL = pd.read_json('FittsLaw18.57377.json')
# filename = 'Head1.json'
# filename = 'Eye1.json'
filename = 'Algorithm2.json'
HL = pd.read_json(filename)
for c in HL.columns:
    if type(HL[c][0]) == type(HL['head_position'][0]):
        parsed = pd.json_normalize(HL[c])
        for co in parsed.columns:
            parsed.rename(columns={co: c + '_' + co}, inplace=True)
        HL = pd.concat([HL, parsed], axis=1)
    else:
        pass

# for c in ['head_position', 'head_rotation', 'head_forward','target_position', 'cursor_rotation',]:
#     parsed = pd.json_normalize(HL[c])
#     for co in parsed.columns:
#         parsed.rename({co: c + co})
#     # parsed.rename()
#     data = pd.concat([HL, parsed], axis=1)
# clean
# HL=pd.read_json('FittsLaw144.5415.json')
# pupil = pd.read_csv('EYE_Compare1626659913.917632.csv')
# eye_list = []
# for row in pupil.itertuples(index=False):
#     python_timestamp = row[1]
#     pupil_data = row[2]
#     json_dict = demjson.decode(pupil_data)
#     # json_dict["python_timestamp"] = python_timestamp
#     # json_dict["timestamp"] = float(json_dict["timestamp"])
#     # json_dict["confidence"] = float(json_dict["confidence"])
#     # json_dict["theta"] = float(json_dict["theta"])
#     # json_dict["phi"] = float(json_dict["phi"])
#     eye_list.append(json_dict)
# pupil = pd.DataFrame(eye_list)
# pupil = pupil[pupil['topic'] == 'pupil.0.3d']

from matplotlib import pyplot as plt
import math
import numpy as np


def angle(v1, v2, acute):
    # v1 is your firsr vector
    # v2 is your second vector

    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle


def target_vector(a, b):
    output = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    norm = np.linalg.norm(output, ord=1)
    return output / norm


data = HL
data['target_vector'] = data.apply(
    lambda row: target_vector([row['head_position_x'], row['head_position_y'], row['head_position_z']],
                              [row['target_position_x'], row['target_position_y'], row['target_position_z']]
                              ), axis=1
)
data['target_vector_x'] = data.apply(
    lambda row: row['target_vector'][0],
    axis=1
)
data['target_vector_y'] = data.apply(
    lambda row: row['target_vector'][1],
    axis=1
)
data['target_vector_z'] = data.apply(
    lambda row: row['target_vector'][2],
    axis=1
)


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle


def vector_to_angle(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    # theta = math.atan2(y, x)

    # p = math.acos(z / r)
    theta = math.atan2(z, x) - math.pi / 2
    p = math.atan2(y, math.sqrt(x * x + z * z))

    return -theta, -p


data['calculated_theta'] = data.apply(
    lambda row: math.degrees(
        change_angle(vector_to_angle(row['head_forward_x'], row['head_forward_y'], row['head_forward_z'])[0])),
    axis=1
)
data['calculated_phi'] = data.apply(
    lambda row: math.degrees(
        change_angle(vector_to_angle(row['head_forward_x'], row['head_forward_y'], row['head_forward_z'])[1])),
    axis=1
)
data['head_rotation_x'] = data['head_rotation_x'].apply(change_angle)
data['head_rotation_y'] = data['head_rotation_y'].apply(change_angle)

# %%

data['eye_only_x'] = data['eyedata_GazeDirection.x'] - data['head_forward_x']
data['eye_only_y'] = data['eyedata_GazeDirection.y'] - data['head_forward_y']
data['eye_only_z'] = data['eyedata_GazeDirection.z'] - data['head_forward_z']

data['eye_xvel'] = data['eyedata_GazeDirection.x'].diff(1) / data['timestamp'].diff(1)
data['eye_yvel'] = data['eyedata_GazeDirection.y'].diff(1) / data['timestamp'].diff(1)
data['eye_vel'] = (data['eye_xvel'] ** 2 + data['eye_yvel'] ** 2).apply(math.sqrt).apply(math.degrees)

data['head_xvel'] = data['head_forward_x'].diff(1) / data['timestamp'].diff(1)
data['head_yvel'] = data['head_forward_y'].diff(1) / data['timestamp'].diff(1)
data['head_vel'] = (data['head_xvel'] ** 2 + data['head_yvel'] ** 2).apply(math.sqrt).apply(math.degrees)

data['cursor_xvel'] = data['cursor_forward_x'].diff(1) / data['timestamp'].diff(1)
data['cursor_yvel'] = data['cursor_forward_y'].diff(1) / data['timestamp'].diff(1)
data['cursor_vel'] = (data['cursor_xvel'] ** 2 + data['cursor_yvel'] ** 2).apply(math.sqrt).apply(math.degrees)
# fig = go.Figure(data=[
#     go.Scatter(x=data.timestamp, y=data.head_vel, name='head_vel'),
#     go.Scatter(x=data.timestamp, y=data.eye_vel, name='eye_vel'),
# ])
# fig.update_layout(title=filename + 'velocity')
# fig.show()
# fig = go.Figure(data=[
#     go.Scatter(x=data.timestamp, y=data['eyedata_GazeDirection.x'], name='gaze'),
#     go.Scatter(x=data.timestamp, y=data['head_forward_x'], name='head'),
#     go.Scatter(x=data.timestamp, y=data['cursor_forward_x'], name='cursor'),
#     go.Scatter(x=data.timestamp, y=data['target_vector_x'], name='target'),
#     # go.Scatter(x=data.timestamp, y=data['eye_only_x'], name='eye only'),
#     # go.Scatter(x=data.timestamp, y=data['easing_x'], name='eased cursor'),
#     # go.Scatter(x=data.timestamp, y=data['easing_x_th'], name='eased cursor_th'),
# ])
# fig.update_layout(title=filename + 'Pitch')
# fig.show()
# fig = go.Figure(data=[
#     go.Scatter(x=data.timestamp, y=data['eyedata_GazeDirection.y'], name='gaze'),
#     go.Scatter(x=data.timestamp, y=data['head_forward_y'], name='head'),
#     go.Scatter(x=data.timestamp, y=data['cursor_forward_y'], name='cursor'),
#     go.Scatter(x=data.timestamp, y=data['target_vector_y'], name='target'),
#     # go.Scatter(x=data.timestamp, y=data['easing_y_th'], name='eased cursor_th'),
# ])
# fig.update_layout(title=filename + 'Yaw')
# fig.show()
head_offsets = []
cursor_offsets = []
head_stds = []
cursor_stds = []
head_IC = []
cursor_IC = []
for start in range(9):
    d = data[data['startNum'] == start]
    d.reset_index(inplace=True)
    d.timestamp = d.timestamp - d.timestamp.values[0]
    d['head_distance'] = (
            (d.target_vector_x - d.head_forward_x) ** 2 + (d.target_vector_y - d.head_forward_y) ** 2).apply(
        math.sqrt)
    d['cursor_distance'] = (
            (d.target_vector_x - d.cursor_forward_x) ** 2 + (d.target_vector_y - d.cursor_forward_y) ** 2).apply(
        math.sqrt)
    initial_contact_head = d[d.head_distance < 0.1].index[0]
    initial_contact_cursor = d[d.cursor_distance < 0.1].index[0]

    head_offset = d[d.index >= initial_contact_head]['head_distance'].mean()
    head_offset_std = d[d.index >= initial_contact_head]['head_distance'].std()
    cursor_offset = d[d.index >= initial_contact_cursor]['cursor_distance'].mean()
    cursor_offset_std = d[d.index >= initial_contact_cursor]['cursor_distance'].std()

    head_offsets.append(head_offset)
    cursor_offsets.append(cursor_offset)
    head_stds.append(head_offset_std)
    cursor_stds.append(cursor_offset_std)
    head_IC.append(d.timestamp[initial_contact_head])
    cursor_IC.append(d.timestamp[initial_contact_cursor])
    print(start, '->', 'initial contact', d.timestamp[initial_contact_head], '->', d.timestamp[initial_contact_cursor],
          d.timestamp[initial_contact_cursor] / d.timestamp[initial_contact_head], 'times')
    print(start, '->', 'mean offset', head_offset, '->', cursor_offset, cursor_offset / head_offset, 'times')
    print(start, '->', 'std offset', head_offset_std, '->', cursor_offset_std, cursor_offset_std / head_offset_std,
          'times')
    print('')
    fig = go.Figure(data=[
        go.Scatter(x=d.timestamp, y=d.head_vel, name='head_vel'),
        go.Scatter(x=d.timestamp, y=d.eye_vel, name='eye_vel'),
        go.Scatter(x=d.timestamp, y=d.cursor_vel, name='cursor_vel'),
        go.Scatter(x=[d.timestamp[initial_contact_head]],y=[0],fillcolor='blue'),
        go.Scatter(x=[d.timestamp[initial_contact_cursor]], y=[0], fillcolor='magenta'),
    ])
    fig.update_layout(title=filename + ' : ' + str(start) + ' velocity')
    fig.show()


def mean(data):
    output = 0
    for i in data:
        output += i
    return output / len(data)


print(filename)
print('initial contact time', mean(head_IC), '->', mean(cursor_IC))
print('mean offset', mean(head_offsets), '->', mean(cursor_offsets))
print('std offset', mean(head_stds), '->', mean(cursor_stds))

# %%
from sklearn.mixture import GaussianMixture

new_data = data['eye_vel'][1:].to_numpy().reshape(-1, 1)
# hx,hy,_ = plt.hist(new_data,bins=100)
# plt.grid()
# plt.show()
gmm = GaussianMixture(n_components=2).fit(new_data)
# gm = GaussianMixture(n_components=2,random_state=0).fit([data['eye_xvel'][1:],data['eye_yvel'][1:]])
# print(gmm.weights_)
print('gmm mean', gmm.means_[0], gmm.means_[1])
print('cov', gmm.covariances_)
print('precision', gmm.precisions_)
#
print('prediction', gmm.predict(new_data))
plt.plot(data['timestamp'][1:], data['eye_vel'][1:])
plt.plot(data['timestamp'][1:], gmm.predict(new_data) * 200)
# plt.plot(data['timestamp'][1:],data['eye_angular_distance'][1:])
plt.show()
# print(gmm.predict([[1]]))
plt.plot(data['timestamp'], data['eye_angular_distance'])
fast = data[data['eye_vel'] > gmm.means_[1][0]].index.tolist()
for t in fast:
    plt.axvline(data['timestamp'][t], color='red', alpha=0.3)
plt.show()
