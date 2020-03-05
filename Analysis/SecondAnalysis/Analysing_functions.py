from OneEuroFilter import *
import pandas as pd

def normalize(dataset):
    # dataset = ((dataset - dataset.mean()) / (dataset.max() - dataset.min()))
    dataset = dataset / (dataset.max() - dataset.min())
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
