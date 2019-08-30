import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.transform import Rotation as R

from fileHandling import *
from quaternionTest import *
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
    # ProcessingData[0]: Pupil data...
    # ProcessingData[1]: IMU data...

    imuTimeStamp = ProcessingData[1]['ImuTimeStamp'].astype(float)
    Quaternions = ProcessingData[1][['QuatReal', 'QuatI', 'QuatJ', 'QuatK']]

    # TODO: quaternion to euler angle ... see quaternionTeset.py
    angleX = []
    angleY = []
    angleZ = []
    for index, row in Quaternions.iterrows():
        r = R.from_quat([row.QuatI, row.QuatJ, row.QuatK, row.QuatReal])
        angles = r.as_euler('zxy', degrees=True)
        # angleX.append(angles[0])
        # angleY.append(angles[1])
        # angleZ.append(angles[2])
        quaternion = quaternion_to_euler(row.QuatI, row.QuatJ, row.QuatK, row.QuatReal)
        angleX.append(quaternion[0])
        angleY.append(quaternion[1])
        angleZ.append(quaternion[2])
    ProcessingData[1]['angleX'] = angleX
    ProcessingData[1]['angleY'] = angleY
    ProcessingData[1]['angleZ'] = angleZ
    # print(ProcessingData[1][['angleX','angleY','angleZ']])

    AngleX = ProcessingData[1]['angleX'].astype(float)
    AngleY = ProcessingData[1]['angleY'].astype(float)
    AngleZ = ProcessingData[1]['angleZ'].astype(float)

    HoloTimeStamp = HololensData[0]['UTC'].astype(float)
    HoloQuaternions = HololensData[0][
        ['HeadQuaternionX', 'HeadQuaternionY', 'HeadQuaternionZ', 'HeadQuaternionW']].astype(float)

    # '''Hololens quaternions'''
    HoloAngleX = []
    HoloAngleY = []
    HoloAngleZ = []
    dcmX = []
    dcmY = []
    dcmZ = []
    for index, row in HoloQuaternions.iterrows():
        r = R.from_quat([row.HeadQuaternionW, row.HeadQuaternionX, row.HeadQuaternionY, row.HeadQuaternionZ])
        angles = r.as_euler('zyx', degrees=True)
        # dcm.append(r.as_dcm())
        a = r.as_dcm()
        # dcmX.append(np.matmul(a, [[1], [0], [0]])[0])
        # dcmY.append(np.matmul(a, [[1], [0], [0]])[1])
        # dcmZ.append(np.matmul(a, [[1], [0], [0]])[2])
        HoloAngleX.append(-angles[0])
        HoloAngleY.append(angles[1])
        HoloAngleZ.append(-angles[2] - 180)
        quatTest = quaternion_to_euler(row.HeadQuaternionX, row.HeadQuaternionY, row.HeadQuaternionZ,
                                       row.HeadQuaternionW)
        dcmX.append(quatTest[0])
        dcmY.append(quatTest[1])
        dcmZ.append(quatTest[2])
    HololensData[0]['dcmX'] = dcmX
    HololensData[0]['dcmY'] = dcmY
    HololensData[0]['dcmZ'] = dcmZ
    # print(HololensData[0]['dcm'])
    HololensData[0]['angleX'] = HoloAngleX
    HololensData[0]['angleY'] = HoloAngleY
    HololensData[0]['angleZ'] = HoloAngleZ

    Conv_HeadAngleX = HololensData[0]['angleX'].astype(float)
    Conv_HeadAngleY = HololensData[0]['angleY'].astype(float)
    Conv_HeadAngleZ = HololensData[0]['angleZ'].astype(float)
    HeadAngleX = HololensData[0]['HeadAngleX'].astype(float)
    HeadAngleY = HololensData[0]['HeadAngleY'].astype(float)
    HeadAngleZ = HololensData[0]['HeadAngleZ'].astype(float)
    # peaks, _ = find_peaks(Conv_HeadAngleX)
    # peaks, _ = find_peaks(Conv_HeadAngleX, height=0)
    # peaks2, _ = find_peaks(pIntersectZ)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.title.set_text('angle')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.title.set_text('quat')

    Conv_HololensAngleXInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, Conv_HeadAngleX, kind='quadratic')
    Conv_HololensAngleYInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, Conv_HeadAngleY, kind='quadratic')
    Conv_HololensAngleZInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, Conv_HeadAngleZ, kind='quadratic')

    HololensAngleXInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleX, kind='quadratic')
    HololensAngleYInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleY, kind='quadratic')
    HololensAngleZInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleZ, kind='quadratic')

    angleXInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleX, kind='quadratic')
    angleYInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleY, kind='quadratic')
    angleZInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleZ, kind='quadratic')
    # 1170 = 6.5s * 180Hz
    HololensInterpolatedTimeStamp = np.linspace(HoloTimeStamp[0], HoloTimeStamp.tail(1).values[0], 1170).squeeze()
    imuInterpolatedTimeStamp = np.linspace(imuTimeStamp[0], imuTimeStamp.tail(1).values[0], 1170)

    Conv_HololensAngleX = Conv_HololensAngleXInterpolatedFunction(HololensInterpolatedTimeStamp)
    Conv_HololensAngleY = Conv_HololensAngleYInterpolatedFunction(HololensInterpolatedTimeStamp)
    Conv_HololensAngleZ = Conv_HololensAngleZInterpolatedFunction(HololensInterpolatedTimeStamp)

    HololensAngleX = HololensAngleXInterpolatedFunction(HololensInterpolatedTimeStamp)
    HololensAngleY = HololensAngleYInterpolatedFunction(HololensInterpolatedTimeStamp)
    HololensAngleZ = HololensAngleZInterpolatedFunction(HololensInterpolatedTimeStamp)

    imuAngleX = angleXInterpolatedFunction(imuInterpolatedTimeStamp)
    imuAngleY = angleYInterpolatedFunction(imuInterpolatedTimeStamp)
    imuAngleZ = angleZInterpolatedFunction(imuInterpolatedTimeStamp)

    ax1.plot(HololensInterpolatedTimeStamp, HololensAngleX, 'r')
    ax1.plot(HololensInterpolatedTimeStamp, HololensAngleY, 'g')
    ax1.plot(HololensInterpolatedTimeStamp, HololensAngleZ, 'b')

    # ax2.plot(HololensInterpolatedTimeStamp, Conv_HololensAngleX, 'r')
    # ax2.plot(HololensInterpolatedTimeStamp, Conv_HololensAngleY, 'g')
    # ax2.plot(HololensInterpolatedTimeStamp, Conv_HololensAngleZ, 'b')
    ax2.plot(imuInterpolatedTimeStamp, imuAngleX, 'r')
    ax2.plot(imuInterpolatedTimeStamp, imuAngleY, 'g')
    # ax2.plot(imuInterpolatedTimeStamp, imuAngleZ, 'b')
    # ax2.plot(HololensData[0]['dcmX'], 'r')
    # ax2.plot(HololensData[0]['dcmY'], 'g')
    # ax2.plot(HololensData[0]['dcmZ'], 'b')
    # ax2.plot(AngleX)
    # ax2.plot(AngleY)
    # ax2.plot(AngleZ)

    # s = phase_align(HololensAngleX, imuAngleX, [0, 500])
    # s = phase_align(f, pIntersectZ, [10, 500])
    # ax2.plot(shift(pIntersectZ, s, mode='nearest'), ls='--', label='aligned data')
    # print('phase shift value to align is ', s)
    # ax2.plot(imuInterpolatedTimeStamp, shift(imuAngleX, s, mode='nearest'), ls='--')
    # Handle IMU values... quaternions.?

    plt.show()


def changeAngle(angle):
    if angle > 180:
        return angle - 360
    else:
        return angle
