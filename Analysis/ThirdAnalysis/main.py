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
from analysing_functions import *

# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = "browser"

# %%bandwidth test

from plotly.subplots import make_subplots
from analysing_functions import *

# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal

pio.renderers.default = "browser"
target = 5
env = "W"
block = 2
subject = 301

holo, imu, eye = bring_data(target, env, block, subject)
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)

eye = eye[eye.confidence > 0.9]
eye.timestamp = eye.timestamp - shift_time
new_holo, new_imu, new_eye = interpolated_dataframes(holo, imu, eye)
new_imu = imu_to_vector(new_imu)
new_holo = holo_to_vector(new_holo)


# new_holo = new_holo[new_holo.timestamp >= 1.5];new_holo.reset_index(inplace=True)
# new_imu = new_imu[new_imu.timestamp >= 1.5];new_imu.reset_index(inplace=True)
# new_eye = new_eye[new_eye.timestamp >= 1.5];new_eye.reset_index(inplace=True)

class algorithm:
    raw = []
    filtered = []

    order = 2
    lowcut = 0.001
    highcut = 0.99

    zi = []

    def __init__(self, order=2, lowcut=0.001, highcut=0.99):
        self.raw = []
        self.filtered = []
        self.order = order,
        self.lowcut = lowcut
        self.highcut = highcut
        self.zi = [0] * 2 * order
        self.b, self.a = signal.iirfilter(order, [self.lowcut, self.highcut],
                                          btype='bandpass', analog=False, ftype='butter')

    def add_data(self, _data):
        self.raw.append(_data)
        self.filter(_data)

    def filter(self, _data):
        res, z = signal.lfilter(self.b, self.a, [_data], zi=self.zi)
        self.zi = z
        self.filtered.append(res[0])

    def get_last(self):
        return self.filtered[-1]

    def reset(self):
        self.raw = []
        self.filtered = []
        self.zi = [0] * 2 * self.order


class overall_algorithm:
    data = {
        ('head', 'H'): algorithm(order=2, lowcut=0.001, highcut=0.99),
        ('head', 'V'): algorithm(order=2, lowcut=0.001, highcut=0.80),
        ('eye', 'H'): algorithm(order=2, lowcut=0.001, highcut=0.99),
        ('eye', 'V'): algorithm(order=2, lowcut=0.001, highcut=0.80),
    }
    offset = 120
    frame_count = 0
    H_multiples = []
    V_multiples = []

    # b_H_head, a_H_head = signal.iirfilter(2, [0.001, 0.99], btype='bandpass', analog=False, ftype='butter')
    # b_H_eye, a_H_eye = signal.iirfilter(2, [0.001, 0.40], btype='bandpass', analog=False, ftype='butter')
    # b_V_head, a_V_head = signal.iirfilter(2, [0.001, 0.99], btype='bandpass', analog=False, ftype='butter')
    # b_V_eye, a_V_eye = signal.iirfilter(2, [0.001, 0.40], btype='bandpass', analog=False, ftype='butter')

    def __init__(self, name, H_offset=120,V_offset=120):
        self.name = name
        self.H_offset = H_offset
        self.V_offset = V_offset

    def add_data(self, H_head, V_head, H_eye, V_eye):
        self.frame_count += 1
        self.data[('head', 'H')].add_data(H_head)
        self.data[('head', 'V')].add_data(V_head)
        self.data[('eye', 'H')].add_data(H_eye)
        self.data[('eye', 'V')].add_data(V_eye)
        # if self.offset < len(self.data[('eye', 'H')].filtered):
        self.calculate_multiple()

    def calculate_multiple(self):
        if self.offset > len(self.data[('eye', 'H')].filtered):
            H_offset = 0
            V_offset = 0
            if len(self.data[('eye', 'H')].filtered) == 1:
                return
        else:
            H_offset = self.H_offset
            V_offset = self.V_offset

        H_multiple, _ = normalize(self.data[('eye', 'H')].filtered[-H_offset:],
                                  self.data[('head', 'H')].filtered[-H_offset:])
        V_multiple, _ = normalize(self.data[('eye', 'V')].filtered[-V_offset:],
                                  self.data[('head', 'V')].filtered[-V_offset:])
        self.H_multiples.append(H_multiple)
        self.V_multiples.append(V_multiple)


simulation = overall_algorithm('test',H_offset=120,V_offset=15)

# new_holo.head_rotation_x =new_holo.head_rotation_x-new_holo.head_rotation_x[0]
# new_holo.head_rotation_y =new_holo.head_rotation_y-new_holo.head_rotation_y[0]
new_eye.norm_x = new_eye.norm_x - new_eye.norm_x[0]
new_eye.norm_y = new_eye.norm_y - new_eye.norm_y[0]
for i in range(new_holo.shape[0]):
    simulation.add_data(new_holo.head_rotation_y[i], new_holo.head_rotation_x[i], new_eye.norm_x[i], new_eye.norm_y[i])

# %%

# H_mults = pd.Series(simulation.H_multiples.copy())
# V_mults = pd.Series(simulation.V_multiples.copy())
# H_mults = H_mults.rolling(window=30,min_periods=1).mean()
# V_mults = V_mults.rolling(window=30,min_periods=1).mean()

plt.plot(new_holo.timestamp[121:], pd.Series(simulation.H_multiples[120:]))
plt.plot(new_holo.timestamp[121:], pd.Series(simulation.V_multiples[120:]))
# plt.plot(new_holo.timestamp[121:], H_mults[120:])
# plt.plot(new_holo.timestamp[121:], V_mults[120:])
plt.show()

mul = [0]
# rol_mul = [0]
for i in range(len(simulation.H_multiples)):
    mul.append(simulation.data[('eye', 'H')].filtered[i + 1] * simulation.H_multiples[i])
    # rol_mul.append(simulation.data[('eye', 'H')].filtered[i + 1] * H_mults[i])
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y=-new_holo.TargetHorizontal, name='Target'),
        # go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'H')].raw, name='raw-head'),
        # go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'H')].filtered, name='filtered-head'),
        go.Scatter(x=new_holo.timestamp, y=mul, name='multiple-eye'),
        go.Scatter(x=new_holo.timestamp, y=pd.Series(mul) + new_holo.TargetHorizontal, name='diff')
        # go.Scatter(x=new_holo.timestamp, y=rol_mul, name='rol_multiple-eye'),

    ]
)
fig.show()

mul = [0]
# rol_mul=[0]
for i in range(len(simulation.V_multiples)):
    mul.append(simulation.data[('eye', 'V')].filtered[i + 1] * simulation.V_multiples[i])
    # rol_mul.append(simulation.data[('eye', 'V')].filtered[i + 1] * V_mults[i])
fig = go.Figure(
    data=[
        go.Scatter(x=new_holo.timestamp, y=-new_holo.TargetVertical, name='Target'),
        go.Scatter(x=new_holo.timestamp, y=-new_holo.TargetVertical.rolling(window=60,min_periods=1).mean(), name='roll_Target'),
        # go.Scatter(x=new_holo.timestamp, y=simulation.data[('head', 'V')].raw, name='raw-head'),
        # go.Scatter(x=new_holo.timestamp, y=-pd.Series(simulation.data[('head', 'V')].filtered), name='filtered-head'),
        # go.Scatter(x=new_holo.timestamp, y=mul, name='multiple-eye'),
        go.Scatter(x=new_holo.timestamp, y=pd.Series(mul) + new_holo.TargetVertical, name='diff'),
        go.Scatter(x=new_holo.timestamp, y=(pd.Series(mul) + new_holo.TargetVertical).rolling(window=60,min_periods=1).mean(), name='rolled+diff'),
        # go.Scatter(x=new_eye.timestamp, y=new_eye.norm_y * 250, name='original-eye'),
        # go.Scatter(x=new_holo.timestamp, y=rol_mul, name='rol_multiple-eye'),

    ]
)
fig.show()
