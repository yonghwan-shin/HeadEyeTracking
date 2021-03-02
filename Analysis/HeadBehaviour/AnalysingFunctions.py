from scipy import signal
import numpy as np
import math
import vg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
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


def filter_sbs(data, cutoff=10, fs=200):
    fc = cutoff
    w = fc / (fs / 2)
    b, a = signal.butter(3, w, 'high')
    result = []
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, data, zi=zi * data[0])
    return z


def winter_low(cutoff_freq, sample_time, x0, x1, x2, y1, y2, print_coeff=False):
    """Filters a data sample based on two past unfiltered and filtered data samples.

    2nd order low pass, single pass butterworth filter presented in Winter2009.

    Parameters
    ==========
    cuttoff_freq: float
        The desired lowpass cutoff frequency in Hertz.
    sample_time: floaat
        The difference in time between the current time and the previous time.
    x0 : float
        The current unfiltered signal, x_i
    x1 : float
        The unfiltered signal at the previous sampling time, x_i-1.
    x2 : float
        The unfiltered signal at the second previous sampling time, x_i-2.
    y1 : float
        The filtered signal at the previous sampling time, y_i-1.
    y2 : float
        The filtered signal at the second previous sampling time, y_i-2.

    """
    sampling_rate = 1 / sample_time  # Hertz

    correction_factor = 1.0  # 1.0 for a single pass filter

    corrected_cutoff_freq = np.tan(np.pi * cutoff_freq / sampling_rate) / correction_factor  # radians

    K1 = np.sqrt(2) * corrected_cutoff_freq
    K2 = corrected_cutoff_freq ** 2

    a0 = K2 / (1 + K1 + K2)
    a1 = 2 * a0
    a2 = a0

    K3 = a1 / K2

    b1 = -a1 + K3
    b2 = 1 - a1 - K3

    if print_coeff:
        print('num:', a0, a1, a2)
        print('dem:', 1.0, -b1, -b2)

    return a0 * x0 + a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2


def murphy_high(cutoff_freq, sample_time, x0, x1, x2, y1, y2, print_coeff=False):
    """
    Parameters
    ==========
    cuttoff_freq: float
        The desired lowpass cutoff frequency in Hertz.
    sample_time: floaat
        The difference in time between the current time and the previous time.
    x0 : float
        The current unfiltered signal, x_i
    x1 : float
        The unfiltered signal at the previous sampling time, x_i-1.
    x2 : float
        The unfiltered signal at the second previous sampling time, x_i-2.
    y1 : float
        The filtered signal at the previous sampling time, y_i-1.
    y2 : float
        The filtered signal at the second previous sampling time, y_i-2.
    """
    sampling_rate = 1 / sample_time  # Hertz

    correction_factor = 1.0

    cutoff_freq = 1 / 2 / sample_time - cutoff_freq  # covert high pass freq to equivalent lowpass freq

    corrected_cutoff_freq = np.tan(np.pi * cutoff_freq / sampling_rate) / correction_factor

    K1 = np.sqrt(2) * corrected_cutoff_freq
    K2 = corrected_cutoff_freq ** 2

    a0 = K2 / (1 + K1 + K2)
    a1 = 2 * a0
    a2 = a0

    K3 = a1 / K2

    b1 = -a1 + K3
    b2 = 1 - a1 - K3

    c0 = a0
    c1 = -a1
    c2 = a2

    d1 = -b1
    d2 = b2

    if print_coeff:
        print('num:', c0, c1, c2)
        print('dem:', 1.0, -d1, -d2)

    return c0 * x0 + c1 * x1 + c2 * x2 + d1 * y1 + d2 * y2


def realtime_highpass(time, sig, cutoff):
    output = [0, 0]
    for i in range(2, len(time)):
        output.append(murphy_high(cutoff,
                                  time[i] - time[i - 1],
                                  sig[i], sig[i - 1], sig[i - 2],
                                  output[i - 1], output[i - 2]))
    return output


def realtime_lowpass(time, sig, cutoff):
    output = [sig[0], sig[1]]
    for i in range(2, len(time)):
        output.append(winter_low(cutoff,
                                 time[i] - time[i - 1],
                                 sig[i], sig[i - 1], sig[i - 2],
                                 output[i - 1], output[i - 2]))
    return output


def head_angle_to_angle_distance(H, V, tH, tV):
    H = H.apply(math.radians)
    tH = tH.apply(math.radians)
    V = V.apply(math.radians)
    tV = tV.apply(math.radians)
    output = V.apply(math.cos) * tV.apply(math.cos)
    output = output + V.apply(math.sin) * tV.apply(math.sin) * (H - tH).apply(math.cos)
    output = output.apply(math.acos)
    output = output.apply(math.degrees)
    return output


def angle_to_vector(H, V):
    H = H.apply(math.radians)
    V = V.apply(math.radians)
    x = V.apply(math.cos) * H.apply(math.sin)
    y = V.apply(math.sin)
    z = V.apply(math.cos) * H.apply(math.cos)
    return x, -y, z


def angle_distance(x, y, z, x1, y1, z1):
    vector1 = np.array([x, y, z])
    vector2 = np.array([x1, y1, z1])
    return vg.angle(vector1, vector2)


def get_cosine(a, b, c, d):
    u = np.array([a, c])
    v = np.array([b, d])
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    # c= np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
    # angle = np.degrees(np.arccos(np.clip(c,-1,1)))
    dot_product = np.dot(u, v)
    angle = np.degrees(np.arccos(dot_product))
    # if angle > 90:
    #     angle = 180-angle
    return angle


def get_angle_between_vectors(Heye, Himu, Veye, Vimu):
    df = pd.DataFrame({'Heye': Heye, 'Himu': Himu, 'Veye': Veye, 'Vimu': Vimu})
    df['angle'] = df.apply(
        lambda x: get_cosine(x['Heye'], x['Himu'], x['Veye'], x['Vimu']),
        axis=1
    )
    df['eye_magnitude'] = df.apply(
        lambda x: np.sqrt(x['Heye'] ** 2 + x['Veye'] ** 2),
        axis=1
    )
    df['imu_magnitude'] = df.apply(
        lambda x: np.sqrt(x['Himu'] ** 2 + x['Vimu'] ** 2),
        axis=1
    )
    return df


def get_new_angular_distance(H, V, data):
    x, y, z = angle_to_vector(H, V)
    # print(x)
    data['X'] = np.array(x)
    data['Y'] = np.array(y)
    data['Z'] = np.array(z)
    data['changed_angular_distance'] = data.apply(
        lambda x: angle_distance(x['X'], x['Y'], x['Z'], (x.target_position_x - x.head_position_x),
                                 (x.target_position_y - x.head_position_y), (x.target_position_z - x.head_position_z)),
        axis=1
    )
    return data['changed_angular_distance']


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff_freq, fs, order=3, real_time=False):
    nyq_freq = fs / 2
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    if real_time == False:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


def compare_horizontal(data, _output):
    Target_error = data - _output.Phi
    if len(Target_error[abs(Target_error) < 0.1]) <= 0:
        target_in_index = Target_error.abs().idxmin()
    else:
        target_in_index = Target_error[abs(Target_error) < 0.1].index[0]
    return _output.timestamp.iloc[target_in_index], Target_error[target_in_index:].mean(), Target_error[
                                                                                           target_in_index:].std()


def compare_vertical(data, _output):
    Target_error = data - _output.Theta
    if len(Target_error[abs(Target_error) < 0.1]) <= 0:
        target_in_index = Target_error.abs().idxmin()
    else:
        target_in_index = Target_error[abs(Target_error) < 0.1].index[0]
    return _output.timestamp.iloc[target_in_index], Target_error[target_in_index:].mean(), Target_error[
                                                                                           target_in_index:].std()


def draw_walk_plot(UI, World, column_name, variables):
    xtick = variables
    simple_xtick = ['UI', 'World']
    width = 0.2
    x = np.arange(len(xtick))
    fig, ax = plt.subplots(figsize=(16, 8))
    height1 = [UI[column_name + '_' + str(x)].mean() for x in xtick]
    height2 = [World[column_name + '_' + str(x)].mean() for x in xtick]
    rect1 = ax.bar(x - width / 2, height1, width, label=simple_xtick[0])
    rect2 = ax.bar(x + width / 2, height2, width, label=simple_xtick[1])
    ax.set_ylabel(column_name)
    ax.set_title(column_name)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick)

    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rect1);
    autolabel(rect2);

    ax.yaxis.grid(True)
    fig.tight_layout()
    # plt.show()


def draw_stand_walk_comparison_plot(stand_UI, stand_World, walk_UI, walk_World, column_name, variables):
    xtick = variables
    simple_xtick = ['stand/UI', 'stand/World', 'walk/UI', 'walk/World']
    width = 0.2
    x = np.arange(len(xtick))
    fig, ax = plt.subplots(figsize=(16, 8))
    height1 = [stand_UI[column_name + '_' + str(x)].mean() for x in xtick]
    height2 = [stand_World[column_name + '_' + str(x)].mean() for x in xtick]
    height3 = [walk_UI[column_name + '_' + str(x)].mean() for x in xtick]
    height4 = [walk_World[column_name + '_' + str(x)].mean() for x in xtick]
    yerr1 = [stand_UI[column_name + '_' + str(x)].std() for x in xtick]
    yerr2 = [stand_World[column_name + '_' + str(x)].std() for x in xtick]
    yerr3 = [walk_UI[column_name + '_' + str(x)].std() for x in xtick]
    yeer4 = [walk_World[column_name + '_' + str(x)].std() for x in xtick]
    rect1 = ax.bar(x - 3 * width / 4, height1, width / 2, label=simple_xtick[0])
    rect2 = ax.bar(x - 1 * width / 4, height2, width / 2, label=simple_xtick[1])
    rect3 = ax.bar(x + 1 * width / 4, height3, width / 2, label=simple_xtick[2])
    rect4 = ax.bar(x + 3 * width / 4, height4, width / 2, label=simple_xtick[3])
    ax.set_ylabel(column_name)
    ax.set_title(column_name)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick)

    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rect1);
    autolabel(rect2);
    autolabel(rect3);
    autolabel(rect4)
    ax.yaxis.grid(True)
    fig.tight_layout()
    plt.show()


# def draw_simple_plot(column_name):
#     simple_xtick = ['stand/UI', 'stand/World', 'walk/UI', 'walk/World']
#     simple_x = [0, 1, 2, 3]
#     height = [stand_UI[column_name].mean(), stand_World[column_name].mean(),
#               walk_UI[column_name].mean(), walk_World[column_name].mean()]
#     yerr = [stand_UI[column_name].std(), stand_World[column_name].std(),
#             walk_UI[column_name].std(), walk_World[column_name].std()]
#     fig, ax = plt.subplots()
#     rect = ax.bar(simple_x, height, yerr=yerr, capsize=10)
#     ax.set_xlabel('dwell threshold (sec)')
#     ax.set_ylabel(column_name)
#     ax.set_title(column_name)
#     ax.set_xticks(simple_x)
#     ax.set_xticklabels(simple_xtick)
#
#     # ax.legend()
#
#     def autolabel(rects):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate('{:.2f}'.format(height),
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom')
#
#     autolabel(rect)
#     ax.yaxis.grid(True)
#     fig.tight_layout()
#     plt.show()

class LowPassFilter(object):
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class OneEuroFilter(object):

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def change_cutoff(self, mincutoff):
        self.__mincutoff = mincutoff

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))


def one_euro(_data, timestamp=None, freq=60, mincutoff=1, beta=1.0, dcutoff=1.0):
    config = dict(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
    filter = OneEuroFilter(**config)
    f = []
    _data = list(_data)
    timestamp = list(timestamp)

    for i in range(len(_data)):
        if timestamp is None:
            f.append(filter(x=_data[i]))
        else:
            f.append(filter(_data[i], timestamp=timestamp[i]))
    return pd.Series(f)


class real_time_peak_detection():
    def __init__(self, array, lag, threshold, influence):
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0, self.avgFilter[i], self.stdFilter[i]
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
            return 0, self.avgFilter[i], self.stdFilter[i]

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i], self.avgFilter[i], self.stdFilter[i]
