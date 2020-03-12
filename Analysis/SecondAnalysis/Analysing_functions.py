from OneEuroFilter import *
import pandas as pd
# from sympy import *
import math

def normalize(dataset, to_dataset):
    # dataset = ((dataset - dataset.mean()) / (dataset.max() - dataset.min()))
    multiple = (to_dataset.max() - to_dataset.min()) / (dataset.max() - dataset.min())
    dataset = multiple * (dataset - dataset.min()) + to_dataset.min()
    return dataset


def eye_one_euro_filtering(x, y):
    config1 = {
        'freq': 120,  # Hz
        'mincutoff': 0.78,  # FIXME
        'beta': 1.0,  # FIXME
        'dcutoff': 1.0  # this one should be ok
    }
    config2 = {
        'freq': 120,  # Hz
        'mincutoff': 0.87,  # FIXME
        'beta': 0.99,  # FIXME
        'dcutoff': 1.0  # this one should be ok
    }
    filtered_x = []
    filtered_y = []
    oneeurox = OneEuroFilter(**config1)
    oneeuroy = OneEuroFilter(**config2)

    for i in range(len(x)):
        filtered_x.append(oneeurox(x[i]))
        filtered_y.append(oneeuroy(y[i]))

    return pd.Series(filtered_x), pd.Series(filtered_y)


def asSpherical(xyz: list):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.acos(z / r) * 180 / math.pi  # to degrees
    phi = math.atan2(y, x) * 180 / math.pi
    return [r, theta, phi]
