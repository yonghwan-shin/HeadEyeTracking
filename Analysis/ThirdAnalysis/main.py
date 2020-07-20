#%% Importing
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
from plotly.subplots import make_subplots

from filehandling import bring_hololens_data, read_eye_file, read_imu_file

#%% Bring Data
subject = 9995
env = "U"
holo = bring_hololens_data(3, env, 1, subject)
imu = read_imu_file(3, env, 1, subject)
eye = read_eye_file(3, env, 1, subject)

# %% draw 3d plot of walking trace
fig = px.line_3d(
    holo,
    x="head_position_x",
    z="head_position_y",
    y="head_position_z",
    range_x=[-0.5, 0.5],
    range_z=[-0.5, 0.5],
    range_y=[0, 8],
    width=600,
    height=600,
)
fig.show()

# %% simple comparison of hololens & IMU record
fig = px.line(holo, x="timestamp", y="head_rotation_x")
fig.show()
fig = px.line(imu, x="IMUtimestamp", y="rotationX")
fig.show()
# %% Filtering...?
""" NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi
"""
# eye = eye[eye.confidence >0.6]
fig = make_subplots(rows=1, cols=2)
fig.add_trace(
    go.Scatter(x=imu.IMUtimestamp, y=imu.rotationX, name="IMU-vertical"), row=1, col=1
)
# fig.add_trace(go.Scatter(x=eye.timestamp,y= eye.theta.rolling(window=20).mean()),row=1,col=2,name='eye-filtered')

b, a = scipy.signal.butter(3, 0.05)
filtered = scipy.signal.filtfilt(b, a, eye.theta)
fig.add_trace(
    go.Scatter(
        x=eye.timestamp, y=filtered, mode="markers", name="eye-filtered-vertical"
    ),
    row=1,
    col=2,
)
fig.show()
# %%
