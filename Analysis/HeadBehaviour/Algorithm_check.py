# %%
from AnalysingFunctions import *

from FileHandling import *
import time
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from natsort import natsorted

sns.set_theme(style='whitegrid')
pio.renderers.default = 'browser'
subject = 1
env = 'U'
target = 3
block = 4
cutoff = 0.5
beta = 0.01

output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                            study_num=3)
if env == 'W':
    r = 0.3 / 2
else:
    r = 0.3 / 2
output['MaximumTargetAngle'] = (r * 1 / output.Distance).apply(math.asin) * 180 / math.pi
output['one_euro' + str(cutoff) + '_' + str(beta)] = get_new_angular_distance(
    pd.Series(one_euro(output.head_rotation_y, output.timestamp, 60, cutoff, beta)),
    pd.Series(one_euro(output.head_rotation_x, output.timestamp, 60, cutoff, beta)),
    output)
eye = read_eye_data(target=target, environment=env, posture='W', block=block, subject=subject,
                    study_num=3)
print(eye.confidence.mean())
eye = eye[eye['confidence'] > 0.8]
imu = read_imu_data(target=target, environment=env, posture='W', block=block, subject=subject,
                    study_num=3)
shift, corr, shift_time = synchronise_timestamp(imu, output, show_plot=False)
eye.timestamp = eye.timestamp - shift_time
imu.timestamp = imu.timestamp - shift_time

# %%

fig = go.Figure(
    data=[
        go.Scatter(x=output.timestamp, y=output.head_rotation_x, name='original'),

        go.Scatter(x=output.timestamp, y=realtime_lowpass(output.timestamp, output.head_rotation_x, cutoff),
                   name='winter_low'),
        # go.Scatter(x=output.timestamp, y=butter_lowpass_filter(output.head_rotation_y,cutoff,60,2,False), name='butter_false'),
        go.Scatter(x=output.timestamp, y=butter_lowpass_filter(output.head_rotation_x, cutoff, 60, 2, True),
                   name='butter_true'),
        go.Scatter(x=output.timestamp, y=one_euro(output.head_rotation_x, output.timestamp, 60, cutoff, 0.01),
                   name='one'),
        go.Scatter(x=output.timestamp, y=output.Theta, name='phi'),

        go.Scatter(x=imu.timestamp, y=imu.rotationX, name='imuz'),
        go.Scatter(x=output.timestamp, y=output.angular_distance, name='angle distance'),
        go.Scatter(x=output.timestamp, y=output.MaximumTargetAngle, name='max'),
        go.Scatter(x=output.timestamp, y=output['one_euro' + str(cutoff) + '_' + str(beta)], name='angle_filtered')
    ]
)
fig.show()

# %%
Timestamp = np.arange(0, 6.5, 1 / 120)
Vholo = interpolate.interp1d(output.timestamp, output.head_rotation_x, fill_value='extrapolate')
Vimu = interpolate.interp1d(imu.timestamp, imu.rotationX, fill_value='extrapolate')
Veye = interpolate.interp1d(eye.timestamp, eye.norm_y, fill_value='extrapolate')
Hholo = interpolate.interp1d(output.timestamp, output.head_rotation_y, fill_value='extrapolate')
Himu = interpolate.interp1d(imu.timestamp, imu.rotationZ, fill_value='extrapolate')
Heye = interpolate.interp1d(eye.timestamp, eye.norm_x, fill_value='extrapolate')
AngleSpeed = interpolate.interp1d(output.timestamp, output.angle_speed, fill_value='extrapolate')

Hpre_cutoff = 5.0
Vpre_cutoff = 3.0
# Vholo = one_euro(pd.Series(Vholo(Timestamp)), Timestamp, 120, Vpre_cutoff, 0.01)
Vholo = pd.Series(Vholo(Timestamp))
# Vimu = one_euro(pd.Series(Vimu(Timestamp)), Timestamp, 120, Vpre_cutoff, 0.01)
Vimu = pd.Series(realtime_lowpass(Timestamp, Vimu(Timestamp), Vpre_cutoff))
# Veye = one_euro(pd.Series(Veye(Timestamp)), Timestamp, 120, Vpre_cutoff, 0.01)
Veye = pd.Series(realtime_lowpass(Timestamp, Veye(Timestamp), Vpre_cutoff))
# Hholo = one_euro(pd.Series(Hholo(Timestamp)), Timestamp, 120, Hpre_cutoff, 0.01)
Hholo= pd.Series(Hholo(Timestamp))
# Himu = one_euro(pd.Series(Himu(Timestamp)), Timestamp, 120, Hpre_cutoff, 0.01)
Himu = pd.Series(realtime_lowpass(Timestamp, Himu(Timestamp), Hpre_cutoff))
# Heye = one_euro(pd.Series(Heye(Timestamp)), Timestamp, 120, Hpre_cutoff, 0.01)
Heye = pd.Series(realtime_lowpass(Timestamp, Heye(Timestamp), Hpre_cutoff))

AngleSpeed = pd.Series(AngleSpeed(Timestamp))
# fig = go.Figure(data=[
#     go.Scatter(x=Timestamp,y=Vholo.diff(1),name='Vholo'),
#     # go.Scatter(x=Timestamp,y=Vimu,name='Vimu'),
#     go.Scatter(x=Timestamp,y=Veye.diff(1)*100,name='Veye')
# ])
# fig.show()

# fig = go.Figure(data=[
#     go.Scatter(x=Timestamp,y=Hholo.diff(1),name='Hholo'),
#     # go.Scatter(x=Timestamp,y=Vimu,name='Vimu'),
#     go.Scatter(x=Timestamp,y=Heye.diff(1)*100,name='Heye')
# ])
vector = (Heye.diff(1) * Himu.diff(1) + Veye.diff(1) * Vimu.diff(1))

fig = go.Figure(data=[
    go.Scatter(x=Timestamp, y=Hholo.diff(1), name='HholoVel'),
    go.Scatter(x=Timestamp, y=Vholo.diff(1), name='VholoVel'),
    go.Scatter(x=Timestamp, y=Vholo, name='Vholo'),
    # go.Scatter(x=Timestamp,y=Veye.diff(1)*Vimu.diff(1),name='Veye'),
    # go.Scatter(x=Timestamp, y=Heye.diff(1), name='HeyeVel'),
    # go.Scatter(x=Timestamp, y=Heye.diff(1) * Himu.diff(1), name='H'),
    # go.Scatter(x=Timestamp, y=Veye.diff(1) * Vimu.diff(1), name='V'),
    # go.Scatter(x=Timestamp, y=Heye, name='Heye'),
    # go.Scatter(x=Timestamp, y=Hholo, name='Hholo'),
    # go.Scatter(x=output.timestamp,y=output.Phi,name/='Phi'),
    go.Scatter(x=Timestamp, y=vector, name='vector'),

    # go.Scatter(x=Timestamp,y=vector.rolling(24,min_periods=1).mean(),name='roll-vector'),
    # go.Scatter(x=Timestamp,y=AngleSpeed,name='Angle speed')
])

fig.update_layout(shapes=[
    dict(
        type='line',
        yref='paper', y0=0, y1=1,
        xref='x', x0=output.timestamp[output[output.target_entered == True].index[0]],
        x1=output.timestamp[output[output.target_entered == True].index[0]]
    )
])
print(output.timestamp[output[output.target_entered == True].index[0]])
fig.show()


# %%
def check_list(lists):
    for i in range(len(lists)):
        if lists[i] <= 0: return False
    return True


def algorithm(Timestamp, Vholo, Vimu, Veye, Hholo, Himu, Heye):
    config_L = dict(freq=120, mincutoff=0.5, beta=0.01, dcutoff=1.0)
    config_H = dict(freq=120, mincutoff=5, beta=0.01, dcutoff=1.0)
    Hfilter = OneEuroFilter(**config_L)
    Vfilter = OneEuroFilter(**config_L)
    _Vholo = list(Vholo)
    _Hholo = list(Hholo)
    vector = list(Heye.diff(1) * Himu.diff(1) + Veye.diff(1) * Vimu.diff(1))
    V = []
    H = []
    end = []
    on = False
    length = 8
    timestamp = list(Timestamp)
    for i in range(len(Vholo)):
        if i > length:
            if on == False:
                on = check_list(vector[i - length:i])
                # for j in range(length):
                #     if vector[i - j] < 0:
                #         on=False;break
                if on == True:
                    print(on, Timestamp[i])
            else:
                if abs(Himu.diff(1).loc[i]) < 0.02 or abs(Vimu.diff(1).loc[i]) < 0.02:
                    on = False
        if on:
            end.append(Timestamp[i])
            Hfilter.change_cutoff(5)
            Vfilter.change_cutoff(5)
        else:
            Hfilter.change_cutoff(0.3)
            Vfilter.change_cutoff(0.3)
        V.append(Vfilter(_Vholo[i]))
        H.append(Hfilter(_Hholo[i]))
    return V, H, end


V, H, end = algorithm(Timestamp, Vholo, Vimu, Veye, Hholo, Himu, Heye)

fig = go.Figure(data=[
    go.Scatter(x=Timestamp, y=V, name='V'),
    go.Scatter(x=Timestamp, y=H, name='H'),
    go.Scatter(x=Timestamp, y=Vholo, name='Vholo'),
    go.Scatter(x=Timestamp, y=Hholo, name='Hholo'),
    go.Scatter(x=output.timestamp, y=output.Phi, name='phi'),
    go.Scatter(x=Timestamp, y=one_euro(Hholo,Timestamp,120,0.3,0.01), name='Hholo-oneeuro'),
])
s = []
for e in end:
    s.append(dict(
        type='line',
        yref='paper', y0=0, y1=1,
        xref='x', x0=e,
        x1=e
    ))
# fig.update_layout(shapes=s)
fig.show()
