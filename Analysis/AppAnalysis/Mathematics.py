import numpy as np
import math

def LinearRegression(x, y):
    if len(x) != len(y):
        raise Exception("different length of x, y")
    if len(x)==1:
        return 0,0,1
    sumOfX = 0
    sumOfY = 0
    sumOfXSq = 0
    sumOfYSq = 0
    sumCodeviates = 0
    for i in range(len(x)):
        sumCodeviates += x[i] * y[i]
        sumOfX += x[i]
        sumOfY += y[i]
        sumOfXSq += x[i] * x[i]
        sumOfYSq += y[i] * y[i]
    count = len(x)
    ssX = sumOfXSq - ((sumOfX * sumOfX) / count)
    ssY = sumOfYSq - ((sumOfY * sumOfY) / count)
    rNumerator = (count * sumCodeviates) - (sumOfX * sumOfY)
    rDenom = (count * sumOfXSq - (sumOfX * sumOfX)) * (count * sumOfYSq - (sumOfY * sumOfY))
    sCo = sumCodeviates - ((sumOfX * sumOfY) / count)
    meanX = sumOfX / count
    meanY = sumOfY / count
    dblR = rNumerator / math.sqrt(rDenom)
    rSquared = dblR * dblR
    yIntercept = meanY - ((sCo / ssX) * meanX)
    slope = sCo / ssX
    return rSquared, yIntercept, slope


def normalize(_from, _to, RMS=False):
    if RMS == True:
        _from = np.array(_from)
        _to = np.array(_to)
        mean_difference = _to.mean() - _from.mean()
        multiple_power2 = np.power((_to - _to.mean()), 2).mean() / np.power((_from - _from.mean()), 2).mean()
        multiple = np.sqrt(multiple_power2)
        return multiple, mean_difference
    a = _from - sum(_from) / len(_from)
    b = _to - sum(_to) / len(_to)
    multiple = (max(b) - min(b)) / (max(a) - min(a))
    shift = sum(_to) / len(_to) - sum(_from) / len(_from)
    return multiple, shift


def change_angle(_angle):
    if _angle > 180:
        _angle = _angle - 360
    return _angle

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
        # self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        # self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
            return 0

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

        return self.signals[i]


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))

def angle_velocity(_head_forward, _head_forward2, _time):
    import vg

    if type(_head_forward2) is not dict:
        return None
    vector1 = np.array([_head_forward["x"], _head_forward["y"], _head_forward["z"]])
    vector2 = np.array([_head_forward2["x"], _head_forward2["y"], _head_forward2["z"]])
    return vg.angle(vector1, vector2) / _time
