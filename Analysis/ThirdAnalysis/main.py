#%% Importing
from filehandling import *
from analysing_functions import *
# %% Filtering...?
""" NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi
"""# %%
holo, imu, eye = bring_data(1, "W", 3, 303)
shift,corr = synchronise_imu(imu, holo)
print(shift,corr)
fig = go.Figure()
fig.add_trace(go.Scatter(x=holo.timestamp, y=holo.head_rotation_x, name='holo'))
fig.add_trace(go.Scatter(x=imu.IMUtimestamp+shift/120, y=imu.rotationX, name='IMU'))
fig.show()

# %%
