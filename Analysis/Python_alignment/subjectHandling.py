import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
from sklearn.decomposition import PCA
from peakdetect import *
from fileHandling import *
from quaternionTest import *
from signal_alignment import *

plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams['lines.markersize'] = 2


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


# def getOneFile(processingFileNameList, hololensFileNameList, subjectNum, fileCount):
def getOneFile(processingFileNameList, hololensFileNameList, subjectNum, info):
	# ProcessingData = getProcessingFile(rootDirectory + processingDirectory +
	#                                    str(getSubject(subjectNum)) + "/" + processingFileNameList[fileCount])
	HololensData = []
	# find holo data
	trialInfo = makeTrialInfo(info)
	for name in processingFileNameList:
		if name[:11] == trialInfo:
			ProcessingData = getProcessingFile(
				rootDirectory + processingDirectory + str(getSubject(subjectNum)) + "/" + name)
			print("Analyzing...  " + name)
			break
	for name in hololensFileNameList:
		if name[:11] == trialInfo:
			HololensData = getHololensFile(rootDirectory + hololensDirectory +
			                               str(getSubject(subjectNum)) + "/" + name)
			break
		else:
			pass
	# handle empty file
	if not HololensData:
		return
	else:
		return [ProcessingData, HololensData, name]


def analyzeFile(ProcessingData, HololensData, filename):
	startTime = time.time()
	if not ProcessingData or not HololensData:
		print('file seems lost...')
		return
	# ProcessingData[0]: Pupil data...
	# ProcessingData[1]: IMU data...
	pupilDataFile = ProcessingData[0]
	imuData = ProcessingData[1]
	imuTimeStamp = imuData['ImuTimeStamp'].astype(float)
	Quaternions = imuData[['QuatReal', 'QuatI', 'QuatJ', 'QuatK']]

	pupilData = pd.DataFrame(columns=['timestamp', 'norm_posX', 'norm_posY', 'confidence'])

	pupilTimeStamp = []
	pupilNorm_PosX = []
	pupilNorm_PosY = []
	pupilConfidence = []
	for i in pupilDataFile:
		pupilTimeStamp.append(i['timestamp'])
		pupilNorm_PosX.append(i['norm_pos'][0])
		pupilNorm_PosY.append(i['norm_pos'][1])
		pupilConfidence.append(i['confidence'])
	pupilData['timestamp'] = pupilTimeStamp
	pupilData['norm_posX'] = pupilNorm_PosX
	pupilData['norm_posY'] = pupilNorm_PosY
	pupilData['confidence'] = pupilConfidence
	# TODO: drop low confidnece values...? -> set threshold as .6
	confidenceThreshold = 0.6
	pupilData = pupilData[pupilData.confidence > confidenceThreshold]
	pupilTimeStamp = pupilData['timestamp']
	pupilNorm_PosX = pupilData['norm_posX']
	pupilNorm_PosY = pupilData['norm_posY']
	pupilConfidence = pupilData['confidence']
	pupilData.drop_duplicates('timestamp')
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
	HeadAngleX = HololensData[0]['HeadAngleX'].astype(float)
	HeadAngleY = HololensData[0]['HeadAngleY'].astype(float)
	HeadAngleZ = HololensData[0]['HeadAngleZ'].astype(float)
	# peaks, _ = find_peaks(Conv_HeadAngleX)

	HololensAngleXInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleX, kind='quadratic')
	HololensAngleYInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleY, kind='quadratic')
	HololensAngleZInterpolatedFunction = interpolate.interp1d(HoloTimeStamp, HeadAngleZ, kind='quadratic')
	angleXInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleX, kind='quadratic')
	angleYInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleY, kind='quadratic')
	angleZInterpolatedFunction = interpolate.interp1d(imuTimeStamp, AngleZ, kind='quadratic')
	pupilXInterpolatedFunction = interpolate.interp1d(pupilTimeStamp, pupilNorm_PosX, kind='quadratic')
	pupilYInterpolatedFunction = interpolate.interp1d(pupilTimeStamp, pupilNorm_PosY, kind='quadratic')
	pupilConfidenceInterpolatedFunction = interpolate.interp1d(pupilTimeStamp, pupilConfidence, kind='quadratic')

	# 1170 = 6.5s * 180Hz
	HololensInterpolatedTimeStamp = np.linspace(HoloTimeStamp[0], HoloTimeStamp.tail(1).values[0], 1170).squeeze()
	imuInterpolatedTimeStamp = np.linspace(imuTimeStamp[0], imuTimeStamp.tail(1).values[0], 1170)
	pupilInterpolatedTimeStamp = np.linspace(pupilTimeStamp[0], pupilTimeStamp.tail(1).values[0], 1170)

	HololensAngleX = HololensAngleXInterpolatedFunction(HololensInterpolatedTimeStamp)
	HololensAngleY = HololensAngleYInterpolatedFunction(HololensInterpolatedTimeStamp)
	HololensAngleZ = HololensAngleZInterpolatedFunction(HololensInterpolatedTimeStamp)

	imuAngleX = angleXInterpolatedFunction(imuInterpolatedTimeStamp)
	imuAngleY = angleYInterpolatedFunction(imuInterpolatedTimeStamp)
	imuAngleZ = angleZInterpolatedFunction(imuInterpolatedTimeStamp)

	PupilNormX = pupilXInterpolatedFunction(pupilInterpolatedTimeStamp)
	PupilNormY = pupilYInterpolatedFunction(pupilInterpolatedTimeStamp)
	PupilConfidence = pupilConfidenceInterpolatedFunction(pupilInterpolatedTimeStamp)

	# TODO: find pupil positions' peak and see motion vector arrangements
	# TODO: try this with original pupil data...
	fig = plt.figure(figsize=(7, 7))
	# ax1 = fig.add_subplot(2, 1, 1)
	# ax2 = fig.add_subplot(2, 1, 2)

	Xpeaks = peakdetect(PupilNormX, lookahead=15)
	Xpeaksmax = Xpeaks[0]
	Xpeaksmin = Xpeaks[1]
	Xpeakmax = [Xpeaksmax[i][0] for i in range(len(Xpeaksmax))]
	Xpeakmin = [Xpeaksmin[i][0] for i in range(len(Xpeaksmin))]

	# ax1.plot(pupilInterpolatedTimeStamp, PupilNormX)
	# ax1.plot(pupilInterpolatedTimeStamp[Xpeakmax], PupilNormX[Xpeakmax], 'rx', markersize=10)
	# ax1.plot(pupilInterpolatedTimeStamp[Xpeakmin], PupilNormX[Xpeakmin], 'kx', markersize=10)
	Ypeaks = peakdetect(PupilNormY, lookahead=15)
	Ypeaksmax = Ypeaks[0]
	Ypeaksmin = Ypeaks[1]
	Ypeakmax = [Ypeaksmax[i][0] for i in range(len(Ypeaksmax))]
	Ypeakmin = [Ypeaksmin[i][0] for i in range(len(Ypeaksmin))]
	# ax2.plot(pupilInterpolatedTimeStamp, PupilNormY)
	# ax2.plot(pupilInterpolatedTimeStamp[Ypeakmax], PupilNormY[Ypeakmax], 'rx', markersize=10)
	# ax2.plot(pupilInterpolatedTimeStamp[Ypeakmin], PupilNormY[Ypeakmin], 'kx', markersize=10)

	Xpeakmax += Xpeakmin
	Xpeakmax += Ypeakmax
	Xpeakmax += Ypeakmin
	finalPeaks = list(set(Xpeakmax))
	finalPeaks.sort()
	Xpeakpoints = [PupilNormX[0]]
	Xpeakpoints += PupilNormX[finalPeaks]
	Xpeakpoints += [PupilNormX[-1]]

	Ypeakpoints = [PupilNormY[0]]
	Ypeakpoints += PupilNormY[finalPeaks]
	Ypeakpoints += [PupilNormY[-1]]

	Xvector = np.diff(Xpeakpoints)
	Yvector = np.diff(Ypeakpoints)
	vectors = np.column_stack((Xvector, Yvector))
	for row in vectors:
		plt.plot(row[0] * 100, row[1] * 100, 'ro', markersize=4)
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.axis('equal')
	# Vector END#############################################################################\

	# ax1 = fig.add_subplot(3, 1, 1)
	# ax2 = fig.add_subplot(3, 1, 2)
	# ax3 = fig.add_subplot(3, 1, 3)
	# ax1.legend(loc='right')
	# ax2.legend(loc='right')
	# ax3.legend(loc='right')
	# ax1.title.set_text('Normal X')
	# ax2.title.set_text('Normal Y')
	# ax3.title.set_text('confidence')
	#
	#   # x -> x , y -> z , z -> y
	# ax3.plot(HololensInterpolatedTimeStamp, HololensAngleX, 'k', label='hololens Angle X', ls='--')
	# ax1.plot(HololensInterpolatedTimeStamp, HololensAngleY, 'g',label='hololens Angle Y')
	# ax1.plot(HololensInterpolatedTimeStamp, HololensAngleZ, 'b', label='hololens Angle Z')
	#
	# ax2.plot(imuInterpolatedTimeStamp, imuAngleX, 'r', label='imu Angle X', ls='--')
	# ax2.plot(imuInterpolatedTimeStamp, imuAngleY, 'g', label='imu Angle Y')
	# ax2.plot(imuInterpolatedTimeStamp, imuAngleZ, 'b',label='imu Angle Z')
	#
	# angleX_shift = phase_align(HololensAngleX, imuAngleX, [0, 500])
	# angleX_shift = phase_align(imuAngleX, HololensAngleX, [200, 800])
	# angleY_shift = phase_align(imuAngleY, HololensAngleZ, [200, 800])
	# angleZ_shift = phase_align(imuAngleZ, HololensAngleY, [200, 800])
	#
	# if angleX_shift < 0 or angleY_shift < 0 or angleZ_shift < 0:
	#     print('phase shift wrong,,, check again, value is ', angleX_shift, angleY_shift, angleZ_shift)
	# else:
	#     print('phase shift value to align is ', angleX_shift, angleY_shift, angleZ_shift)
	# ax3.plot(HololensInterpolatedTimeStamp, shift(HololensAngleX, angleX_shift, mode='nearest'), 'bo')
	# ax3.plot(imuInterpolatedTimeStamp, shift(imuAngleX, -angleX_shift, mode='nearest'),ls='--')
	# ax2.plot(imuInterpolatedTimeStamp, shift(imuAngleY, -angleY_shift, mode='nearest'), ls='--')
	# # ax2.plot(imuInterpolatedTimeStamp, shift(imuAngleZ, -angleZ_shift, mode='nearest'))
	#
	# ax1.plot(pupilInterpolatedTimeStamp, PupilNormX, 'r', label='Pupil Position X')
	# ax2.plot(pupilInterpolatedTimeStamp, PupilNormY, 'b', label='Pupil Position Y')
	# ax3.plot(pupilInterpolatedTimeStamp, PupilConfidence, 'g', label='Confidence')
	# ax1.plot(pupilTimeStamp, pupilNorm_PosX, 'r', label='Pupil norm pos X')
	# ax2.plot(pupilTimeStamp, pupilNorm_PosY, 'b', label='Pupil norm pos Y')
	# ax3.plot(pupilTimeStamp, pupilConfidence, 'g', label='Pupil Confidence')
	#
	# plt.plot(pupilNorm_PosX,pupilNorm_PosY,'x')
	# plt.plot(pupilNorm_PosX[250:650],pupilNorm_PosY[250:650],'x')

	plt.title(filename)

	##############PCA######################
	# ff = np.vstack([pupilNorm_PosX[150:], pupilNorm_PosY[150:]]).T
	# pca1 = PCA(n_components=2)
	# X_low = pca1.fit_transform(ff)
	# X2 = pca1.inverse_transform(X_low)
	# origin = [pca1.mean_[0]], [pca1.mean_[1]]
	# eigen_value = pca1.explained_variance_
	# variance_ratio = pca1.explained_variance_ratio_
	# print(variance_ratio)
	# # draw PCA axis
	#
	# ax = sns.scatterplot(0, 1, data=pd.DataFrame(ff), color='0.2')
	# plt.plot(X2[:,0],X2[:,1],"o-")
	# plt.quiver(*origin, pca1.components_.T[:, 0], pca1.components_.T[:, 1],
	#            color=['r', 'k'], scale=2)
	# # plt.plot(X2[:,0],X2[:,1],"o-",markersize=2)
	# originText = "(" + "{0:.2f}".format(float(origin[0][0])) + "," + "{0:.2f}".format(float(origin[1][0])) + ")"
	# plt.text(x=origin[0][0] + 0.01, y=origin[1][0], s=originText, fontsize=15)
	# plt.axis("equal")
	##############PCA######################
	plt.show()


# print((time.time() - startTime), ' second passed while analyzing both file')


def changeAngle(angle):
	if angle > 180:
		return angle - 360
	else:
		return angle
