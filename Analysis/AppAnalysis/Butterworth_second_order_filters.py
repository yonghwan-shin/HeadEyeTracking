import numpy as np
def butterworth_lowpass(cutoff_freq, sample_time, x0, x1, x2, y1, y2, print_coeff=False):
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


def butterworth_highpass(cutoff_freq, sample_time, x0, x1, x2, y1, y2, print_coeff=False):
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
