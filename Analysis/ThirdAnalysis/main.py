# %% Importing
from analysing_functions import *

""" NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi
"""
# %% test one trial
holo, imu, eye = bring_data(1, "W", 3, 303)
shift, corr,shift_time = synchronise_imu(imu, holo,show_plot=True)
imu.IMUtimestamp = imu.IMUtimestamp - shift_time

fig = go.Figure()
fig.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_x, name='holo'))
fig.add_trace(go.Scatter(x=imu.IMUtimestamp, y=imu.rotationX, name='IMU_original'))
# fig.add_trace(go.Scatter(x=imu.IMUtimestamp+shift/120, y=imu.rotationX, name='IMU_shifted'))
fig.show()


# %%
subjects = range(301, 302)
envs = ["W", "U"]
targets = range(3)
blocks = range(2,3)
# subjects = range(304, 305)
# envs = ["W"]
# targets = range(0, 1)
# blocks = range(3, 4)
for subject in subjects:
    for env, target, block in itertools.product(envs, targets, blocks):
        try:
            print("-" * 10, target, env, block, subject, "-" * 10)
            holo, imu, eye = bring_data(target, env, block, subject)
            shift, corr,shift_time = synchronise_imu(imu, holo)

            print(shift, shift / 200, corr)
        except Exception as e:
            print(e.args)
