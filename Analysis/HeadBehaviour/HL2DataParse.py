# %%
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle


def triangular_weighted_mean(data, window):
    output = []
    full = len(data)
    for i, d in enumerate(data):
        if i < window:
            output.append(data[i])
        else:
            weighted = 0
            div=0
            for j in range(1, window + 1):
                weighted += data[i - window + j] * j
                div+=j
            output.append(weighted/div)

    return output


data = pd.read_json('FittsLaw81.74221.json')

# for col, item in itertools.product(["head_position", "head_rotation", "head_forward", "target_position",'cursor_rotation'],
#                                    ["x", "y", "z"],) :
#     data[col+"_"+item] = data[col].apply(pd.Series)[item]
# for col in ['cursor_rotation_x','cursor_rotation_y','cursor_rotation_z']:
#     data[col] = data[col].apply(change_angle)

eyeData = pd.json_normalize(data.eyedata)
data = pd.concat([data, eyeData], axis=1)
# {'GazeOrigin': {'x': -0.0116990162, 'y': 0.0265332926, 'z': -0.07288933}, 'GazeDirection': {'x': 0.21068461200000002, 'y': -0.1333103, 'z': 0.9684216}, 'calibrationStatus': True, 'EyeTrackingEnabled': True, 'EyeTrackingDataValid': True, 'HeadMovementDirection': {'x': -0.13728515800000002, 'y': 0.33914017700000004, 'z': -0.9306639}, 'HeadVelocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'LatestEyeGazeDirection': {'x': 0.21068461200000002, 'y': -0.1333103, 'z': 0.9684216}, 'EyeTimestamp': '6/24/2021 5:05:56 AM', 'TargetObjectName': 'Target_3'}

plt.plot(data["GazeDirection.x"])
plt.plot(triangular_weighted_mean(data["GazeDirection.x"], 30))
plt.show()

#%%
