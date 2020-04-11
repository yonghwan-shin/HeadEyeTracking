from OneEuroFilter import *
import pandas as pd
# from sympy import *
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import time


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("Working time[{}] : {} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


def parse_timeline(start, end):
    output = []
    if len(start) == 0:
        return [[0.1, 6.3]]
    for i in range(len(start)):
        if i == 0:
            output.append([0.1, start[i]])
        else:
            output.append([end[i - 1], start[i]])
    if len(start) > 0:
        output.append([end[-1], 6.3])
    return output


def normalize(dataset, to_dataset):
    # dataset = ((dataset - dataset.mean()) / (dataset.max() - dataset.min()))
    multiple = (to_dataset.max() - to_dataset.min()) / (dataset.max() - dataset.min())
    # print('multiple is',multiple)
    dataset = multiple * (dataset - dataset.min()) + to_dataset.min()
    return dataset


def shifting(dataset, to_dataset):
    shift = (dataset[0] - to_dataset[0])
    dataset = dataset - shift
    return dataset


def centralise(_dataset):
    _dataset = _dataset - _dataset[0]
    return _dataset


def one_euro(_data, freq=200, mincutoff=3, beta=0.98, dcutoff=1.0):
    config = {
        'freq': freq,  # Hz
        'mincutoff': mincutoff,  # FIXME
        'beta': beta,  # FIXME
        'dcutoff': dcutoff  # this one should be ok
    }
    filter = OneEuroFilter(**config)
    f = []
    for i in range(len(_data)):
        f.append(filter(_data[i]))
    # filtered_data = filter(_data)
    return pd.Series(f)


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


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]
