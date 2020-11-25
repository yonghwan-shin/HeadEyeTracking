# %%

from plotly.subplots import make_subplots
from analysing_functions import *
from IIRfilter import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal, stats

pio.renderers.default = "browser"

target = 2
env = "W"
block = 3
subject = 311
## Bring the result into pandas dataframe
holo, imu, eye = bring_data(target, env, block, subject)
## Get the delayed time between hololens - laptop
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)

## filter out the low-confidene eye data
# eye = eye[eye.confidence > 0.8]

## match the delayed timestamp into hololens' timestamp
eye.timestamp = eye.timestamp - shift_time
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
new_holo = new_holo[new_holo.timestamp > 1.5]
new_eye = new_eye[new_eye.timestamp > 1.5]
new_eye=new_eye[new_eye.confidence>0.9]
# %%
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y= new_holo.TargetHorizontal),
        go.Scatter(x=new_eye.timestamp, y= (new_eye.norm_x-new_eye.norm_x.mean())*250)
        # go.Scatter(x=new_holo.TargetHorizontal, y=new_eye.norm_x, mode='markers')
    ]
)
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y= new_holo.TargetVertical),
        go.Scatter(x=new_eye.timestamp, y= (new_eye.norm_y-new_eye.norm_y.mean())*250)
        # go.Scatter(x=new_holo.TargetHorizontal, y=new_eye.norm_x, mode='markers')
    ]
)
fig.update_layout(title="Pointing error - eye position correlation (yaw)",
                  xaxis_title="Pointing error",
                  yaxis_title="Pupil position",
                  legend_title="Legend Title", )
fig.show()
