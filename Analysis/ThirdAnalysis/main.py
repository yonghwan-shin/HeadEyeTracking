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
import plotly.express as px
import plotly.graph_objects as go
from analysing_functions import (synchronise_timestamp, bring_data, itertools,
                                 go, plt, interpolate, pd, signal)
from filehandling import asSpherical
# IF you are using Pycharm
import plotly.io as pio

pio.renderers.default = 'browser'

# %% test one trial - horizontal
holo, imu, eye = bring_data(2, "W", 3, 316)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)
imu.IMUtimestamp = imu.IMUtimestamp - shift_time
eye.timestamp = eye.timestamp - shift_time
eye.norm_x = eye.norm_x - eye.norm_x.mean()
thetas = []
phis = []
for index, row in holo.iterrows():
    x = row["target_position_x"] - row["head_position_x"]
    y = row["target_position_y"] - row["head_position_y"]
    z = row["target_position_z"] - row["head_position_z"]
    [r, theta, phi] = asSpherical([x, z, y])
    thetas.append(90 - theta)
    phis.append(90 - phi)
holo['Theta'] = thetas
holo["Phi"] = phis
holo['TargetVertical'] = holo.head_rotation_x + holo.Theta
holo['TargetHorizontal'] = holo.head_rotation_y - holo.Phi

# eye=eye[eye.confidence>0.6]
intp_holo = interpolate.interp1d(holo.timestamp, holo.head_rotation_y, fill_value='extrapolate')
intp_imu = interpolate.interp1d(imu.IMUtimestamp, imu.rotationZ, fill_value='extrapolate')
intp_target = interpolate.interp1d(holo.timestamp,holo.TargetHorizontal,fill_value='extrapolate')
holo_y = pd.Series(intp_holo(eye.timestamp))
imu_y = pd.Series(intp_imu(eye.timestamp))
target_hor = pd.Series(intp_target(eye.timestamp))

mult = imu.rotationZ.diff().abs().mean() / eye.norm_x.diff().abs().mean()
fig = make_subplots(rows=4, cols=1)
# EYE
fig.add_trace(go.Scatter(x=eye.timestamp, y=(eye.norm_x * mult*5), name='eye-x'), row=1, col=1)
# HOLO
fig.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_y, name='holo-x'), row=2, col=1)
fig.add_trace(go.Scatter(x=holo.timestamp, y=holo.Phi, name='target-x'), row=2, col=1)
#TARGET
fig.add_trace(go.Scatter(x=holo.timestamp,y=holo.TargetHorizontal,name='diff-x'),row=3,col=1)
# COMPENSATED
fig.add_trace(go.Scatter(x=eye.timestamp,y=target_hor+eye.norm_x*mult*5,name='compensated'),row=4,col=1)

fig.show()
# %%
# subjects = range(301, 302)
# envs = ["W", "U"]
# targets = range(3)
# blocks = range(2, 3)
# # subjects = range(304, 305)
# # envs = ["W"]
# # targets = range(0, 1)
# # blocks = range(3, 4)
# for subject in subjects:
#     for env, target, block in itertools.product(envs, targets, blocks):
#         try:
#             pass
#         except Exception as e:
#             print(e.args)
