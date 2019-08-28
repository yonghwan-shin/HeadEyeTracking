from scipy import interpolate

from fileHandling import *
from signal_alignment import *



def getOneSubject(subjectNum):
    processingFileNameList = getFileNameList(
        rootDirectory, processingDirectory, subjectNum)
    hololensFileNameList = getFileNameList(
        rootDirectory, hololensDirectory, subjectNum)
    print('Fetching files from subject ', subjectNum, '...')
    print('in subject', subjectNum, 'there are total', len(processingFileNameList), 'processing files and',
          len(hololensFileNameList),
          'hololens files')
    return [processingFileNameList, hololensFileNameList]


def getOneFile(processingFileNameList, hololensFileNameList, subjectNum, fileCount):
    print("Analyzing...  " + processingFileNameList[fileCount])
    ProcessingData = getProcessingFile(rootDirectory + processingDirectory +
                                       str(getSubject(subjectNum)) + "/" + processingFileNameList[fileCount])
    HololensData = []
    # find holo data
    for name in hololensFileNameList:
        trialDetail = getTrialInfo(processingFileNameList[fileCount])
        if name[:14] == makeTrialInfo(trialDetail):
            HololensData = getHololensFile(rootDirectory + hololensDirectory +
                                           str(getSubject(subjectNum)) + "/" + name)
            break
        else:
            pass
    # TODO : handle empty file error
    if not HololensData:
        return
    else:
        return [ProcessingData, HololensData]


def analyzeFile(ProcessingData, HololensData):
    if not ProcessingData or not HololensData:
        print('file seems lost...')
        return
    imuTimeStamp = ProcessingData[1]['ImuTimeStamp'].astype(float)
    pIntersectZ = ProcessingData[1]['pIntersectZ'].astype(float)
    QuatI = ProcessingData[1]['QuatI']
    QuatJ = ProcessingData[1]['QuatJ']
    QuatK = ProcessingData[1]['QuatK']
    QuatReal = ProcessingData[1]['QuatReal']

    HoloTimeStamp = HololensData[0]['UTC'].astype(float)

    HeadForwardVectorZ = HololensData[0]['HeadAngleX'].astype(float)
    # peaks, _ = find_peaks(HeadForwardVectorZ)
    # peaks, _ = find_peaks(HeadForwardVectorZ, height=0)
    # peaks2, _ = find_peaks(pIntersectZ)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.title.set_text('original')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.title.set_text('shifted')

    HololensInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadForwardVectorZ, kind='quadratic')
    imuInterpolatedFunction = interpolate.interp1d(imuTimeStamp, pIntersectZ, kind='quadratic')

    HololensInterpolatedTimeStamp = np.linspace(HoloTimeStamp[0], HoloTimeStamp.tail(1).values[0], 1170)
    imuInterpolatedTimeStamp = np.linspace(imuTimeStamp[0], imuTimeStamp.tail(1).values[0], 1170)

    headNew = HololensInterpolatedFunction(HololensInterpolatedTimeStamp)
    imuNew = imuInterpolatedFunction(imuInterpolatedTimeStamp)

    ax1.plot(HololensInterpolatedTimeStamp, headNew)
    # ax2.plot(HololensInterpolatedTimeStamp, headNew)
    ax2.plot(imuInterpolatedTimeStamp, imuNew)

    s = phase_align(headNew, imuNew, [0, 500])
    # s = phase_align(f, pIntersectZ, [10, 500])
    # ax2.plot(shift(pIntersectZ, s, mode='nearest'), ls='--', label='aligned data')
    print('phase shift value to align is ', s)
    ax2.plot(imuInterpolatedTimeStamp, shift(imuNew, s, mode='nearest'), ls='--')
    # Handle IMU values... quaternions.?
    # r = R.from_quat([QuatReal[0], QuatI[0], QuatJ[0], QuatK[0]])
    # print(r.as_euler('zxy', degrees=True))

    plt.show()


def changeAngle(angle):
    if angle > 180:
        return angle - 360
    else:
        return angle
