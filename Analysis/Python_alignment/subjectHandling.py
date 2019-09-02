import matplotlib.pyplot as plt
from scipy import interpolate

from fileHandling import *
from quaternionTest import *
from signal_alignment import *

plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams["axes.grid"] = True


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
    pupilDataFile = ProcessingData[0]
    imuData = ProcessingData[1]
    imuTimeStamp = imuData['ImuTimeStamp'].astype(float)
    Quaternions = imuData[['QuatReal', 'QuatI', 'QuatJ', 'QuatK']]

    # TODO: perpare pupil data -> make pandas.dataFrame...
    pupilData = pd.DataFrame(columns=['timestamp', 'norm_posX', 'norm_posY', 'confidence'])

    pupilTimeStamp = []
    pupilNorm_PosX= []
    pupilNorm_PosY = []
    pupilConfidence = []
    for i in pupilDataFile:
        pupilTimeStamp.append(i['timestamp'])
        pupilNorm_PosX.append(i['norm_pos'][0])
        pupilNorm_PosY.append(i['norm_pos'][1])
        pupilConfidence.append(i['confidence'])
    # pupilDataFrame = pupilDataFrame.append({'timestamp': i['timestamp']}, {'norm_pos': i['norm_pos'][0]},
    #                       {'confidence': i['confidence']})
    # print(i['timestamp'] , i['norm_pos'] , i['confidence'])
    pupilData['timestamp'] = pupilTimeStamp
    pupilData['norm_posX'] = pupilNorm_PosX
    pupilData['norm_posY'] = pupilNorm_PosY
    pupilData['confidence'] = pupilConfidence

    # TODO: quaternion to euler angle ... see quaternionTeset.py : DONE
    angleX = []
    angleY = []
    angleZ = []
    for index, row in Quaternions.iterrows():
        r = R.from_quat([row.QuatI, row.QuatJ, row.QuatK, row.QuatReal])
        quaternion = quaternion_to_euler(row.QuatI, row.QuatJ, row.QuatK, row.QuatReal)
        angleX.append(-quaternion[0])
        angleY.append(-quaternion[1])
        angleZ.append(-90 - quaternion[2])

    imuData['angleX'] = angleX
    imuData['angleY'] = angleY
    imuData['angleZ'] = angleZ

    AngleX = imuData['angleX'].astype(float)
    AngleY = imuData['angleY'].astype(float)
    AngleZ = imuData['angleZ'].astype(float)

    HoloTimeStamp = HololensData[0]['UTC'].astype(float)
    HoloQuaternions = HololensData[0][
        ['HeadQuaternionX', 'HeadQuaternionY', 'HeadQuaternionZ', 'HeadQuaternionW']].astype(float)

    HeadAngleX = HololensData[0]['HeadAngleX'].astype(float)
    HeadAngleY = HololensData[0]['HeadAngleY'].astype(float)
    HeadAngleZ = HololensData[0]['HeadAngleZ'].astype(float)
    # peaks, _ = find_peaks(Conv_HeadAngleX)
    # peaks, _ = find_peaks(Conv_HeadAngleX, height=0)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.title.set_text('Hololens angle X')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.title.set_text('imu angle X')

    HololensAngleXInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleX, kind='quadratic')
    HololensAngleYInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleY, kind='quadratic')
    HololensAngleZInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleZ, kind='quadratic')

    angleXInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleX, kind='quadratic')
    angleYInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleY, kind='quadratic')
    angleZInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleZ, kind='quadratic')
    # 1170 = 6.5s * 180Hz
    HololensInterpolatedTimeStamp = np.linspace(HoloTimeStamp[0], HoloTimeStamp.tail(1).values[0], 1170).squeeze()
    imuInterpolatedTimeStamp = np.linspace(imuTimeStamp[0], imuTimeStamp.tail(1).values[0], 1170)

    HololensAngleX = HololensAngleXInterpolatedFunction(HololensInterpolatedTimeStamp)
    HololensAngleY = HololensAngleYInterpolatedFunction(HololensInterpolatedTimeStamp)
    HololensAngleZ = HololensAngleZInterpolatedFunction(HololensInterpolatedTimeStamp)

    imuAngleX = angleXInterpolatedFunction(imuInterpolatedTimeStamp)
    imuAngleY = angleYInterpolatedFunction(imuInterpolatedTimeStamp)
    imuAngleZ = angleZInterpolatedFunction(imuInterpolatedTimeStamp)

    #   x -> x
    #   y -> z
    #   z -> y
    # ax1.plot(HololensInterpolatedTimeStamp, HololensAngleX, 'b', label='hololens Angle X')
    # ax1.plot(HololensInterpolatedTimeStamp, HololensAngleY, 'g',label='hololens Angle Y')
    # ax1.plot(HololensInterpolatedTimeStamp, HololensAngleZ, 'b', label='hololens Angle Z')

    # ax2.plot(imuInterpolatedTimeStamp, imuAngleX, 'r', label='imu Angle X', ls='--')
    # ax2.plot(imuInterpolatedTimeStamp, imuAngleY, 'g', label='imu Angle Y')
    # ax2.plot(imuInterpolatedTimeStamp, imuAngleZ, 'b',label='imu Angle Z')

    # angleX_shift = phase_align(HololensAngleX, imuAngleX, [0, 500])
    # angleX_shift = phase_align(imuAngleX, HololensAngleX, [200, 800])
    # angleY_shift = phase_align(imuAngleY, HololensAngleZ, [200, 800])
    # angleZ_shift = phase_align(imuAngleZ, HololensAngleY, [200, 800])
    # if angleX_shift < 0 or angleY_shift < 0 or angleZ_shift < 0:
    #     print('phase shift wrong,,, check again, value is ', angleX_shift, angleY_shift, angleZ_shift)
    # else:
    #     print('phase shift value to align is ', angleX_shift, angleY_shift, angleZ_shift)
    #
    # # ax2.plot(imuInterpolatedTimeStamp, shift(imuAngleX, -angleX_shift, mode='nearest'))
    # ax2.plot(imuInterpolatedTimeStamp, shift(imuAngleY, -angleY_shift, mode='nearest'), ls='--')
    # # ax2.plot(imuInterpolatedTimeStamp, shift(imuAngleZ, -angleZ_shift, mode='nearest'))

    # ax1.legend(loc='right')
    # ax2.legend(loc='right')
    ax1.plot(pupilData['norm_posX'])
    ax2.plot(pupilData['norm_posY'])
    # ax1.plot(pupilData['norm_posX'],pupilData['norm_posY'])
    plt.show()


def changeAngle(angle):
    if angle > 180:
        return angle - 360
    else:
        return angle
