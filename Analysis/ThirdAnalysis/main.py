#%% Importing
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
import itertools
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from filehandling import (
    bring_hololens_data,
    read_eye_file,
    read_imu_file,
    filter_out_eye,
)
from scipy import interpolate

#%% Bring Data
def bring_data(target, env, block, subject):
    try:
        holo = bring_hololens_data(target, env, block, subject)
        imu = read_imu_file(target, env, block, subject)
        eye = read_eye_file(target, env, block, subject)
        return holo, imu, eye
    except:
        print("error in bringing data")


# %% draw 3d plot of walking trace
def draw_3d_passage(holo):
    fig = px.scatter_3d(
        holo,
        x="head_position_x",
        z="head_position_y",
        y="head_position_z",
        range_x=[-0.5, 0.5],
        range_z=[-0.5, 0.5],
        range_y=[0, 8],
        width=600,
        height=600,
        color="target_entered",
        opacity=0.5,
    )
    fig.update_traces(marker=dict(size=5))

    fig.show()


# %% simple comparison of hololens & IMU record
def compare_holo_IMU(holo, imu):
    fig = px.line(holo, x="timestamp", y="head_rotation_x")
    fig.show()
    fig = px.line(imu, x="IMUtimestamp", y="rotationX")
    fig.show()


# %% Filtering...?
""" NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi
"""


def filter_visualise(eye):
    eye = eye[eye.confidence > 0.6]
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scatter(x=imu.IMUtimestamp, y=imu.rotationX, name="IMU-vertical"),
        row=1,
        col=1,
    )
    # fig.add_trace(go.Scatter(x=eye.timestamp,y= eye.theta.rolling(window=20).mean()),row=1,col=2,name='eye-filtered')

    b, a = scipy.signal.butter(3, 0.05)
    filtered = scipy.signal.filtfilt(b, a, eye.theta)
    fig.add_trace(
        go.Scatter(
            x=eye.timestamp, y=filtered, mode="markers", name="eye-filtered-vertical"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=eye.timestamp, y=eye.theta, name="eye-raw"), row=2, col=1
    )
    fig.show()


# %%
if __name__ == "__main__":

    subjects = range(301, 302)
    envs = ["W", "U"]
    targets = range(8)
    blocks = range(5)
    # subjects = range(304, 305)
    # envs = ["W"]
    # targets = range(0, 1)
    # blocks = range(3, 4)
    for subject in subjects:
        lowcount = 0
        shortcount = 0
        errcount = 0
        for env, target, block in itertools.product(envs, targets, blocks):
            try:
                # print("-" * 10, target, env, block, subject, "-" * 10)
                eye = read_eye_file(target, env, block, subject)
                comment = filter_out_eye(eye)

            except Exception as e:
                print(e.args)
                if e.args[1] == "short":
                    shortcount = shortcount + 1
                elif e.args[1] == "low":
                    lowcount = lowcount + 1
                else:
                    errcount = errcount + 1
        print(f"{subject} -> err: {errcount}\tshort: {shortcount}\tlow: {lowcount}")


# %%
holo, imu, eye = bring_data(1, "W", 1, 303)

# %%
