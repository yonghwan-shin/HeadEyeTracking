import math


class IIRFilterImplementation:
    z = []

    def __init__(self, order):
        self.z = [0] * order

    def compute(self, input, a):
        result = input
        for t in range(len(a)):
            result += a[t] * self.z[t]
        for t in [len(a) - c - 1 for c in range(len(a))]:
            if t > 0:
                self.z[t] = self.z[t - 1]
            else:
                self.z[t] = result
        return result


class FIRFilterImplementation:
    z = []

    def __init__(self, order):
        self.z = [0] * order

    def compute(self, input, a):
        result = 0
        for t in [len(a) - c - 1 for c in range(len(a))]:
            if t > 0:
                self.z[t] = self.z[t - 1]
            else:
                self.z[t] = input
            result += a[t] * self.z[t]
        return result


class LowpassFilterButterworthSection:
    firFilter = FIRFilterImplementation(3)
    iirFilter = IIRFilterImplementation(2)
    a = []
    b = []
    gain = 0

    def __init__(self, cutoffFrequencyHz, k, n, Fs):
        omegac = 2.0 * Fs * math.tan(math.pi * cutoffFrequencyHz / Fs)
        zeta = -math.cos(math.pi * (2.0 * k + n - 1.0) / (2.0 * n))
        self.a = []
        self.a.append(omegac * omegac)
        self.a.append(2.0 * omegac * omegac)
        self.a.append(omegac * omegac)

        b0 = (4.0 * Fs * Fs) + (4.0 * Fs * zeta * omegac) + (omegac * omegac)
        self.b = []
        self.b.append(
            ((2.0 * omegac * omegac) - (8.0 * Fs * Fs)) / (-b0)
        )
        self.b.append(
            ((4.0 * Fs * Fs) -
             (4.0 * Fs * zeta * omegac) + (omegac * omegac)) / (-b0)
        )
        self.gain = 1.0 / b0

    def compute(self, input):
        return self.iirFilter.compute(
            self.firFilter.compute(self.gain * input, self.a), self.b)


class LowpassFilterButterworthImplementation:
    section = []

    def __init__(self, cutoffFrequencyHz, numSections, Fs):
        self.section = []
        for i in range(numSections):
            self.section.append(LowpassFilterButterworthSection(
                cutoffFrequencyHz, i + 1, numSections * 2, Fs
            ))

    def compute(self, input):
        output = input
        for i in range(len(self.section)):
            output = self.section[i].compute(output)
        return output


class HighpassFilterButterworthSection:
    firFilter = FIRFilterImplementation(3)
    iirFilter = IIRFilterImplementation(2)
    a = []
    b = []
    gain = 0

    def __init__(self, cutoffFrequencyHz, k, n, Fs):
        omegac = 1.0 / (2.0 * Fs * math.tan(math.pi * cutoffFrequencyHz / Fs))
        zeta = -math.cos(math.pi * (2.0 * k + n - 1.0) / (2.0 * n))
        self.a = []
        self.a.append(4.0 * Fs * Fs)
        self.a.append(-8.0 * Fs * Fs)
        self.a.append(4.0 * Fs * Fs)
        b0 = (4.0 * Fs * Fs) + (4.0 * Fs * zeta / omegac) + (1.0 / (omegac * omegac))
        self.b = []
        self.b.append(
            ((2.0 / (omegac * omegac)) - (8.0 * Fs * Fs)) / (-b0)
        )
        self.b.append(
            ((4.0 * Fs * Fs) - (4.0 * Fs * zeta / omegac) + (1.0 / (omegac * omegac))) / (-b0)
        )
        self.gain = 1.0 / b0

    def compute(self, input):
        return self.iirFilter.compute(
            self.firFilter.compute(self.gain * input, self.a), self.b
        )


class HighpassFilterButterworthImplementation:
    section = []

    def __init__(self, cutoffFrequencyHz, numSections, Fs):
        self.section = []
        for i in range(numSections):
            self.section.append(HighpassFilterButterworthSection(
                cutoffFrequencyHz, i + 1, numSections * 2, Fs
            ))

    def compute(self, input):
        output = input
        for section in self.section:
            # for i in range(len(self.section)):
            output = section.compute(output)
        return output


class BandpassFilterButterworthImplementation:

    def __init__(self, bottomFrequencyHz, topFrequencyHz, numSections, Fs):
        self.lowpassFilter = LowpassFilterButterworthImplementation(topFrequencyHz, numSections, Fs)
        self.highpassFilter = HighpassFilterButterworthImplementation(bottomFrequencyHz, numSections, Fs)

    def compute(self, input):
        return self.highpassFilter.compute(self.lowpassFilter.compute(input))

# if __name__=='__main__':
# lpfilter = LowpassFilterButterworthImplementation(20,2,60)
