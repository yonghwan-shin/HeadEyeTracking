# %%

from plotly.subplots import make_subplots
from analysing_functions import *
from IIRfilter import *
# IF you are using Pycharm
import plotly.io as pio
import matplotlib.pyplot as plt
from scipy import fftpack, signal, stats

pio.renderers.default = "browser"

target = 2
env = "U"
block = 2
subject = 301
## Bring the result into pandas dataframe
holo, imu, eye = bring_data(target, env, block, subject)
initial_target_in_time = holo.timestamp[holo[holo.target_entered==True].index[0]]

## Get the delayed time between hololens - laptop
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)


## filter out the low-confidene eye data
# eye = eye[eye.confidence > 0.8]

## match the delayed timestamp into hololens' timestamp
eye.timestamp = eye.timestamp - shift_time
eye_confidence_threshold=0.9
total_eye_frames= eye.shape[0]
eye=eye[eye.confidence>eye_confidence_threshold]
good_eye_frames= eye.shape[0]
good_eye_rate = (100*good_eye_frames/total_eye_frames)
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
print('eye frames', total_eye_frames,'/', 6.5*120,'->',good_eye_frames,' rate:',"{:.2f}".format(good_eye_rate),"%")
if good_eye_rate < 80:
    print('low confidence eye-data')

# Horizontal_head_filter = Butterworth_LowPassFilter(cutoff_freq=5.0)
# Horizontal_eye_filter = Butterworth_LowPassFilter(cutoff_freq=5.0)
#
# horizontal_output=[new_holo.head_rotation_y.iloc[0],new_holo.head_rotation_y.iloc[1]]
# cutoffs=[5,5]
# eye_H=[new_eye.norm_x.iloc[0],new_eye.norm_x.iloc[1]]
#
# Horizontal_head_filter.set_initial_data(new_holo.head_rotation_y.iloc[0],new_holo.head_rotation_y.iloc[1])
# Horizontal_eye_filter.set_initial_data(new_eye.norm_x.iloc[0],new_eye.norm_x.iloc[1])
# change_check=[]
#
#
# for i in range(2,new_eye.shape[0]):
#     Horizontal_head_filter.filter(1/120,new_holo.head_rotation_y.iloc[i])
#     Horizontal_eye_filter.filter(1/120,new_eye.norm_x.iloc[i])
#     horizontal_head_velocity = Horizontal_head_filter.currentValue - Horizontal_head_filter.secondPreviousValue
#     horizontal_eye_velocity = Horizontal_eye_filter.currentValue - Horizontal_eye_filter.secondPreviousValue
#     # print(horizontal_head_velocity,horizontal_eye_velocity)
#     if horizontal_eye_velocity * horizontal_head_velocity <=0:
#         # print(horizontal_head_velocity,horizontal_eye_velocity)
#         change_check.append(True)
#     else:
#         change_check.append(False)
#         # pass
#         # Horizontal_head_filter.set_cutoff_frequency(1.0)
#         # Horizontal_head_filter.set_cutoff_frequency(5.0)
#     if len(change_check)>=10:
#         for i in range(10):
#             if change_check[-i]==False:
#                 Horizontal_head_filter.set_cutoff_frequency(5.0)
#                 break
#             Horizontal_head_filter.set_cutoff_frequency(0.5)
#
#     horizontal_output.append(Horizontal_head_filter.currentValue)
#     cutoffs.append(Horizontal_head_filter.cutoffFrequency)
#     eye_H.append(Horizontal_eye_filter.currentValue)

Horizontal_head_filter_final = OneEuroFilter(120,0.1,0.01,1.0)
Horizontal_head_filter = OneEuroFilter(120,0.5,1,1.0)
Horizontal_eye_filter = OneEuroFilter(120,0.5,1,1.0)
horizontal_head_output=[]
horizontal_eye_output=[]
horizontal_output_final=[]
horizontal_head_velocities=[]
horizontal_eye_velocities=[]
for i in range(new_eye.shape[0]):
    horizontal_head_output.append(Horizontal_head_filter(x=new_holo.head_rotation_y.iloc[i], timestamp=new_holo.timestamp.iloc[i]))
    horizontal_eye_output.append(Horizontal_eye_filter(x=new_eye.norm_x.iloc[i],timestamp=new_eye.timestamp.iloc[i]))

    if i >=1:
        vel_head = (horizontal_head_output[-1] - horizontal_head_output[-2])*120
        vel_eye =( horizontal_eye_output[-1]-horizontal_eye_output[-2])*120
        horizontal_head_velocities.append(vel_head)
        horizontal_eye_velocities.append(vel_eye*200)
        if (vel_eye*vel_head >=0) and abs(vel_head) > 5:

            Horizontal_head_filter_final.change_parameters(0.5, 1)
        else:
            Horizontal_head_filter_final.change_parameters(0.1, 0.01)
        # Horizontal_head_filter_final.change_parameters(0.1, 0.01)
    horizontal_output_final.append(Horizontal_head_filter_final(horizontal_head_output[-1]))
plt.plot(new_holo.timestamp,new_holo.head_rotation_y)
plt.plot(new_holo.timestamp,new_holo.Phi)
plt.plot(new_holo.timestamp, horizontal_head_output)
plt.plot(new_holo.timestamp,horizontal_output_final)
plt.plot(new_holo.timestamp,one_euro(new_holo.head_rotation_y,new_holo.timestamp,120,0.1,0.01,1.0))
# plt.scatter(new_holo.timestamp,cutoffs)
plt.show()
plt.plot(new_eye.timestamp[1:],np.multiply(np.array(horizontal_eye_velocities),np.array(horizontal_head_velocities)))
plt.show()
plt.plot(new_eye.timestamp[1:],horizontal_head_velocities)
plt.plot(new_eye.timestamp[1:],horizontal_eye_velocities)
plt.show()



#%%
# new_holo = new_holo[new_holo.timestamp > initial_target_in_time]
# new_eye= new_eye[new_eye.timestamp>initial_target_in_time]
new_eye.norm_x = new_eye.norm_x - new_eye.norm_x.mean()
new_eye.norm_y = new_eye.norm_y - new_eye.norm_y.mean()
def tocosine(theta):
    return math.cos(math.radians(-theta))
def tosine(theta):
    return math.sin(math.radians(-theta))
new_holo['cos'] = new_holo.head_rotation_z.apply(tocosine)
new_holo['sin'] = new_holo.head_rotation_z.apply(tosine)
new_eye['new_norm_x'] = new_eye.norm_x*new_holo.cos - new_eye.norm_y * new_holo.sin
new_eye['new_norm_y'] = new_eye.norm_x*new_holo.sin + new_eye.norm_y * new_holo.cos
# new_holo.cos = new_holo.cos.apply(math.cos)
# slope_H,intercept_H = linear_regression(new_eye.norm_x-new_eye.norm_x.mean(),new_holo.TargetHorizontal)
# plt.plot(new_eye.timestamp,(new_eye.norm_x-new_eye.norm_x.mean())*slope_H+intercept_H)
# plt.plot(new_holo.timestamp,new_holo.TargetHorizontal)
# plt.show()
# plt.plot(new_holo.timestamp,new_holo.Phi)
plt.plot(new_holo.timestamp,new_holo.norm_head_vector_x)
plt.plot(new_holo.timestamp,new_holo.norm_target_vector_x)
plt.plot(new_holo.timestamp,new_holo.norm_target_vector_x - new_holo.norm_head_vector_x,label='target diff')
plt.plot(new_eye.timestamp,(new_eye.norm_x-new_eye.norm_x.mean())*3,label='original_norm')
plt.plot(new_eye.timestamp,(new_eye.new_norm_x-new_eye.new_norm_x.mean())*2,label = 'new_norm')
# low_H= butter_lowpass_filter(new_holo.head_rotation_y,0.3,120,2,False)
# plt.plot(new_holo.timestamp,low_H)
plt.legend()
plt.show()
plt.plot(new_holo.timestamp,new_holo.norm_head_vector_y)
plt.plot(new_holo.timestamp,new_holo.norm_target_vector_y)
plt.plot(new_holo.timestamp,-new_holo.norm_target_vector_y + new_holo.norm_head_vector_y,label='target diff')
plt.plot(new_eye.timestamp,(new_eye.norm_y-new_eye.norm_y.mean())*3,label='original_norm')
plt.plot(new_eye.timestamp,(new_eye.new_norm_y-new_eye.new_norm_y.mean())*2,label = 'new_norm')
plt.legend()
plt.show()

# %%
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y= new_holo.norm_head_vector_x),
        go.Scatter(x=new_holo.timestamp, y= new_holo.norm_target_vector_x),
go.Scatter(x=new_holo.timestamp, y= new_holo.norm_target_vector_x-new_holo.norm_head_vector_x),
        go.Scatter(x=new_eye.timestamp, y=(new_eye.new_norm_x - new_eye.new_norm_x.mean()) * 2),
        go.Scatter(x=new_eye.timestamp, y=(new_eye.norm_x - new_eye.norm_x.mean()) * 3)
        # go.Scatter(x=new_holo.TargetHorizontal, y=new_eye.norm_x, mode='markers')
    ]
)
# fig = go.Figure(
#     data=[
#         # go.Scatter(x=new_holo.timestamp, y= -new_holo.TargetVertical),
#         go.Scatter(x=new_eye.timestamp, y= (new_eye.new_norm_y-new_eye.new_norm_y.mean())*100),
#         go.Scatter(x=new_eye.timestamp, y= (new_eye.norm_y-new_eye.norm_y.mean())*100)
#         # go.Scatter(x=new_holo.TargetHorizontal, y=new_eye.norm_x, mode='markers')
#     ]
# )
fig.update_layout(title="Pointing error - eye position correlation (yaw)",
                  xaxis_title="Pointing error",
                  yaxis_title="Pupil position",
                  legend_title="Legend Title", )
fig.show()
