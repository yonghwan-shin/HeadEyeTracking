# %%
import pandas as pd
import demjson
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

# HL = pd.read_json('FittsLaw40.25813.json')
# noisy
HL = pd.read_json('FittsLaw18.57377.json')

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
data['eye_only_x'] = data['eyedata_GazeDirection.x'] - data['head_forward_x']
data['eye_only_y'] = data['eyedata_GazeDirection.y'] - data['head_forward_y']
data['eye_only_z'] = data['eyedata_GazeDirection.z'] - data['head_forward_z']

data['eye_xvel'] = data['eyedata_GazeDirection.x'].diff(1) / data['timestamp'].diff(1)
data['eye_yvel'] = data['eyedata_GazeDirection.y'].diff(1) / data['timestamp'].diff(1)
data['eye_vel'] = (data['eye_xvel'] ** 2 + data['eye_yvel'] ** 2).apply(math.sqrt).apply(math.degrees)

data['head_xvel'] = data['head_forward_x'].diff(1) / data['timestamp'].diff(1)
data['head_yvel'] = data['head_forward_y'].diff(1) / data['timestamp'].diff(1)
data['head_vel'] = (data['head_xvel'] ** 2 + data['head_yvel'] ** 2).apply(math.sqrt).apply(math.degrees)

# easing to head
easing_param = 0.1
data['easing_x'] = data['head_forward_x'] * easing_param + data['eyedata_GazeDirection.x'] * (1 - easing_param)
eased_cursor_x = []
eased_cursor_y = []
sac_threshold = 200
for index, row in data.iterrows():
    if index == 0:
        eased_cursor_x.append(row['head_forward_x'])
        eased_cursor_y.append(row['head_forward_y'])
    else:
        # set threshold to 400
        easing_param_temp = max(min(1, row['eye_vel'] / sac_threshold), 0.3)
        new_cursor_x = row['head_forward_x'] * easing_param_temp + row['eyedata_GazeDirection.x'] * (
                1 - easing_param_temp)
        new_cursor_y = row['head_forward_y'] * easing_param_temp + row['eyedata_GazeDirection.y'] * (
                1 - easing_param_temp)
        new_cursor_x = new_cursor_x * easing_param + eased_cursor_x[-1] * (1 - easing_param)
        new_cursor_y = new_cursor_y * easing_param + eased_cursor_y[-1] * (1 - easing_param)
        eased_cursor_x.append(new_cursor_y)
        eased_cursor_y.append(new_cursor_y)
data['easing_x_th'] = eased_cursor_x
data['easing_y_th'] = eased_cursor_y

fig = go.Figure(data=[
    go.Scatter(x=data.timestamp, y=data.head_vel, name='head_vel'),
    go.Scatter(x=data.timestamp, y=data.eye_vel, name='eye_vel'),
])
fig.update_layout(title='velocity')
fig.show()
fig = go.Figure(data=[
    go.Scatter(x=data.timestamp, y=data['eyedata_GazeDirection.x'], name='gaze'),
    go.Scatter(x=data.timestamp, y=data['head_forward_x'], name='head'),
    go.Scatter(x=data.timestamp, y=data['target_vector_x'], name='target'),
    go.Scatter(x=data.timestamp, y=data['eye_only_x'], name='eye only'),
    go.Scatter(x=data.timestamp, y=data['easing_x'], name='eased cursor'),
    go.Scatter(x=data.timestamp, y=data['easing_x_th'], name='eased cursor_th'),
])
fig.update_layout(title='Yaw')
fig.show()
fig = go.Figure(data=[
    go.Scatter(x=data.timestamp, y=data['eyedata_GazeDirection.y'], name='gaze'),
    go.Scatter(x=data.timestamp, y=data['head_forward_y'], name='head'),
    go.Scatter(x=data.timestamp, y=data['target_vector_y'], name='target'),
    go.Scatter(x=data.timestamp, y=data['easing_y_th'], name='eased cursor_th'),
])
fig.update_layout(title='Pitch')
fig.show()
# %%
from sklearn.mixture import GaussianMixture

new_data = data['eye_vel'][1:].to_numpy().reshape(-1, 1)
# hx,hy,_ = plt.hist(new_data,bins=100)
# plt.grid()
# plt.show()
gmm = GaussianMixture(n_components=2).fit(new_data)
# gm = GaussianMixture(n_components=2,random_state=0).fit([data['eye_xvel'][1:],data['eye_yvel'][1:]])
# print(gmm.weights_)
print(gmm.means_[0], gmm.means_[1])
print(gmm.covariances_)
print(gmm.precisions_)
#
print(gmm.predict(new_data))
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
