import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
from sklearn.decomposition import PCA
from peakdetect import *
from fileHandling import *

from signal_alignment import *

plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams['lines.markersize'] = 2


def get_one_subject(subjectNum):
	processingFileNameList = get_filename_list(
		ROOT_DIRECTORY, PROCESSING_DIRECTORY, subjectNum)
	hololensFileNameList = get_filename_list(
		ROOT_DIRECTORY, HOLOLENS_DIRECTORY, subjectNum)
	print(f'Fetching files from subject {subjectNum} ', end='-> ')
	print(
		f'there are total {len(processingFileNameList)} processing files and {len(hololensFileNameList)} hololens files')
	return [processingFileNameList, hololensFileNameList]


# def getOneFile(processingFileNameList, hololensFileNameList, subjectNum, fileCount):
def get_each_file(processingFileNameList, hololensFileNameList, subjectNum, info):
	# ProcessingData = getProcessingFile(rootDirectory + processingDirectory +
	#                                    str(getSubject(subjectNum)) + "/" + processingFileNameList[fileCount])
	HololensData = []
	ProcessingData = []
	# find holo data
	trialInfo = make_trial_info(info)
	for name in processingFileNameList:
		if name[:11] == trialInfo:
			ProcessingData = get_processing_file(
				ROOT_DIRECTORY + PROCESSING_DIRECTORY + str(get_subject(subjectNum)) + "/" + name)
			print(f"Getting file: '{name}'...")
			break
	for name in hololensFileNameList:
		if name[:11] == trialInfo:
			HololensData = get_hololens_file(ROOT_DIRECTORY + HOLOLENS_DIRECTORY +
			                                 str(get_subject(subjectNum)) + "/" + name)
			break
		else:
			pass
	# handle empty file
	if not HololensData:
		print(f"no Hololens Data for {trialInfo}")
		raise ValueError("hololens data is empty")
	elif not ProcessingData:
		print(f"no Processing Data for {trialInfo}")
		raise ValueError("processing data is empty")
	return [ProcessingData, HololensData, trialInfo]
	# return [ProcessingData, HololensData, name]

def filter_files(ProcessingData,HololensData,filename):

	[pupil_data, imu_data] = ProcessingData
	pupil_dataframe = organise_pupil_data(pupil_data)
	imu_dataframe = organise_imu_data(imu_data)

	currentDirectory = os.getcwd()
	filePath = os.path.join(currentDirectory,"refined_data")
	if os.path.exists(filePath):
		pass
	else:
		os.mkdir(filePath)
	pupil_dataframe.to_csv(filePath +"/pupil_"+ str(filename)+".csv", index=False)
	imu_dataframe.to_csv(filePath +"/imu_"+str(filename)+".csv",index=False)



def lookup_file(ProcessingData, HololensData, filename):
	startTime = time.time()
	[pupil_data, imu_data] = ProcessingData

	# get eye values from pupil-cam
	# [pupil_timestamp, pupil_norm_pos_x, pupil_norm_pos_y, pupil_confidence,pupil_theta,pupil_phi] = organise_pupil_data(pupil_data_file)
	pupil_dataframe = organise_pupil_data(pupil_data)
	pupil_timestamp = pupil_dataframe['timestamp']
	pupil_norm_pos_x = pupil_dataframe['norm_posX']
	pupil_norm_pos_y = pupil_dataframe['norm_posY']
	pupil_confidence = pupil_dataframe['confidence']
	pupil_theta = pupil_dataframe['theta']
	pupil_phi = pupil_dataframe['phi']
	if pupil_timestamp is None:
		return None
	if pupil_timestamp.size <100:
		print(filename, 'pupil data loss')
		return None

	# pupil_timestamp = pupil_timestamp * 1000
	# get angle values from IMU
	imu_dataframe = organise_imu_data(imu_data)
	imu_timestamp = imu_dataframe['ImuTimeStamp'].astype(float)
	imu_angle_x = imu_dataframe['angleX'].astype(float)
	imu_angle_y = imu_dataframe['angleY'].astype(float)
	imu_angle_z = imu_dataframe['angleZ'].astype(float)
	# [imu_angle_x, imu_angle_y, imu_angle_z, imu_timestamp] = organise_imu_data(imu_data)
	# get Hololnes data
	hololens_timestamp = HololensData[0]['UTC'].astype(float)
	hololens_angle_x = HololensData[0]['HeadAngleX'].astype(float)
	hololens_angle_y = HololensData[0]['HeadAngleY'].astype(float)
	hololens_angle_z = HololensData[0]['HeadAngleZ'].astype(float)

	# peaks, _ = find_peaks(Conv_HeadAngleX)
	# Interpolations...
	# interpolate_hololens_angle_x = interpolate.interp1d(hololens_timestamp, hololens_angle_x, kind='quadratic')
	# interpolate_hololens_angle_y = interpolate.interp1d(hololens_timestamp, hololens_angle_y, kind='quadratic')
	# interpolate_hololens_angle_z = interpolate.interp1d(hololens_timestamp, hololens_angle_z, kind='quadratic')
	interpolate_imu_angle_x = interpolate.interp1d(imu_timestamp, imu_angle_x, kind='quadratic')
	interpolate_imu_angle_y = interpolate.interp1d(imu_timestamp, imu_angle_y, kind='quadratic')
	interpolate_imu_angle_z = interpolate.interp1d(imu_timestamp, imu_angle_z, kind='quadratic')
	interpolate_pupil_x = interpolate.interp1d(pupil_timestamp, pupil_norm_pos_x, kind='quadratic')
	interpolate_pupil_y = interpolate.interp1d(pupil_timestamp, pupil_norm_pos_y, kind='quadratic')
	# interpolate_pupil_confidence = interpolate.interp1d(pupil_timestamp, pupil_confidence, kind='quadratic')
	interpolate_pupil_theta = interpolate.interp1d(pupil_timestamp, pupil_theta, kind='quadratic')
	interpolate_pupil_phi = interpolate.interp1d(pupil_timestamp, pupil_phi, kind='quadratic')


	# fix timestamp count into 1170 (= 6.5s * 180Hz)
	hololens_transformed_timestamp = np.linspace(hololens_timestamp.iloc[0], hololens_timestamp.tail(1).values[0],
	                                             1170).squeeze()
	imu_transformed_timestamp = np.linspace(imu_timestamp.iloc[0], imu_timestamp.tail(1).values[0], 1170)

	pupil_transformed_timestamp = np.linspace(pupil_timestamp.iloc[0], pupil_timestamp.tail(1).values[0], 1170).squeeze()

	# Calculate new interpolated-points including proper timestamp
	# transformed_hololens_angle_x = interpolate_hololens_angle_x(hololens_transformed_timestamp)
	# transformed_hololens_angle_y = interpolate_hololens_angle_y(hololens_transformed_timestamp)
	# transformed_hololens_angle_z = interpolate_hololens_angle_z(hololens_transformed_timestamp)

	transformed_imu_angle_x = interpolate_imu_angle_x(imu_transformed_timestamp[270:])
	transformed_imu_angle_y = interpolate_imu_angle_y(imu_transformed_timestamp[270:])
	transformed_imu_angle_z = interpolate_imu_angle_z(imu_transformed_timestamp[270:])

	transformed_pupil_norm_x = interpolate_pupil_x(pupil_transformed_timestamp[270:])
	transformed_pupil_norm_y = interpolate_pupil_y(pupil_transformed_timestamp[270:])
	# transformed_pupil_confidence = interpolate_pupil_confidence(pupil_transformed_timestamp)
	transformed_pupil_theta = interpolate_pupil_theta(pupil_transformed_timestamp)
	transformed_pupil_phi = interpolate_pupil_phi(pupil_transformed_timestamp)


	# return np.array([np.max(transformed_imu_angle_z) - np.min(transformed_imu_angle_z),
	#                  np.max(transformed_imu_angle_x) - np.min(transformed_imu_angle_x),
	#                  np.max(transformed_pupil_norm_x) - np.min(transformed_pupil_norm_x),
	#                  np.max(transformed_pupil_norm_y) - np.min(transformed_pupil_norm_y),
	#                  np.max(transformed_pupil_theta) - np.min(transformed_pupil_theta),
	#                  np.max(transformed_pupil_phi) - np.min(transformed_pupil_phi)
	#                  ])

	# print(pupil_transformed_timestamp[:3])
	# print(hololens_transformed_timestamp[:3])
	# print(imu_transformed_timestamp[:3])
	# TODO: see pupil diameter...
	# TODO: try this with original pupil data... <- watch difference between interpolated function and original one
	# fig =plt.figure()
	# ax1 = fig.add_subplot(2, 1, 1)
	# ax2 = fig.add_subplot(2, 1, 2)
	# pca_analysis(transformed_pupil_norm_x, transformed_pupil_norm_y)
	# ax1.plot(pupil_transformed_timestamp[270:],transformed_pupil_phi[270:])
	# ax2.plot(pupil_transformed_timestamp[270:], transformed_pupil_norm_x)
	# ax2.plot(pupil_transformed_timestamp[270:], transformed_pupil_phi[270:])
	# ax1.plot(pupil_transformed_timestamp[270:],transformed_pupil_norm_y,'rx')
	# ax2.plot(imu_transformed_timestamp[270:],transformed_imu_angle_x,'bx')
	# plt.plot(transformed_imu_angle_x,transformed_pupil_norm_y,'rx')
	# plt.plot(transformed_imu_angle_z,transformed_pupil_norm_x,'bx')



	# Alignment

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
	# ax3.plot(hololens_transformed_timestamp, transformed_hololens_angle_x, 'k', label='hololens Angle X', ls='--')
	# ax1.plot(hololens_transformed_timestamp, transformed_hololens_angle_y, 'g',label='hololens Angle Y', ls='--')
	# ax1.plot(hololens_transformed_timestamp, transformed_hololens_angle_z, 'b', label='hololens Angle Z', ls='--')

	# ax1.plot(imu_transformed_timestamp, transformed_imu_angle_x, 'r', label='imu Angle X')
	# ax2.plot(imu_transformed_timestamp, transformed_imu_angle_y, 'g', label='imu Angle Y')
	# ax2.plot(imu_transformed_timestamp, transformed_imu_angle_z, 'b',label='imu Angle Z')

	# angleX_shift = phase_align(transformed_hololens_angle_x, transformed_imu_angle_x, [0, 500])
	# angleX_shift = phase_align(transformed_imu_angle_x, transformed_hololens_angle_x, [200, 800])
	# angleY_shift = phase_align(transformed_imu_angle_y, transformed_hololens_angle_z, [200, 800])
	# angleZ_shift = phase_align(transformed_imu_angle_z, transformed_hololens_angle_y, [200, 800])
	#
	# if angleX_shift < 0 or angleY_shift < 0 or angleZ_shift < 0:
	#     print('phase shift wrong,,, check again, value is ', angleX_shift, angleY_shift, angleZ_shift)
	# else:
	#     print('phase shift value to align is ', angleX_shift, angleY_shift, angleZ_shift)
	# ax3.plot(hololens_transformed_timestamp, shift(transformed_hololens_angle_x, angleX_shift, mode='nearest'), 'bo')
	# ax3.plot(imu_transformed_timestamp, shift(transformed_imu_angle_x, -angleX_shift, mode='nearest'),ls='--')
	# ax2.plot(imu_transformed_timestamp, shift(transformed_imu_angle_y, -angleY_shift, mode='nearest'), ls='--')
	# # ax2.plot(imu_transformed_timestamp, shift(transformed_imu_angle_z, -angleZ_shift, mode='nearest'))
	#
	# ax1.plot(pupil_transformed_timestamp, transformed_pupil_norm_x, 'r', label='Pupil Position X')
	# ax2.plot(pupil_transformed_timestamp, transformed_pupil_norm_y, 'b', label='Pupil Position Y')
	# ax3.plot(pupil_transformed_timestamp, transformed_pupil_confidence, 'g', label='Confidence')
	# ax1.plot(pupil_timestamp, pupil_norm_pos_x, 'r', label='Pupil norm pos X')
	# ax2.plot(pupil_timestamp, pupil_norm_pos_y, 'b', label='Pupil norm pos Y')
	# ax3.plot(pupil_timestamp, pupil_confidence, 'g', label='Pupil Confidence')
	#
	# plt.plot(pupil_norm_pos_x,pupil_norm_pos_y,'x')
	# plt.plot(pupil_norm_pos_x[250:650],pupil_norm_pos_y[250:650],'x')

	# plt.title(filename)
	#
	# plt.show()


# print((time.time() - startTime), ' second passed while analyzing both file')
# def compensation(imu_data,pupil_data):

def pca_analysis(x_list, y_list):
	Xpeaks = peakdetect(x_list, lookahead=15)
	Xpeaksmax = Xpeaks[0]
	Xpeaksmin = Xpeaks[1]
	Xpeakmax = [Xpeaksmax[i][0] for i in range(len(Xpeaksmax))]
	Xpeakmin = [Xpeaksmin[i][0] for i in range(len(Xpeaksmin))]
	# ax1.plot(pupilInterpolatedTimeStamp, PupilNormX)
	# ax1.plot(pupilInterpolatedTimeStamp[Xpeakmax], x_list[Xpeakmax], 'rx', markersize=10)
	# ax1.plot(pupilInterpolatedTimeStamp[Xpeakmin], x_list[Xpeakmin], 'kx', markersize=10)
	Ypeaks = peakdetect(y_list, lookahead=15)
	Ypeaksmax = Ypeaks[0]
	Ypeaksmin = Ypeaks[1]
	Ypeakmax = [Ypeaksmax[i][0] for i in range(len(Ypeaksmax))]
	Ypeakmin = [Ypeaksmin[i][0] for i in range(len(Ypeaksmin))]
	# ax2.plot(pupilInterpolatedTimeStamp, PupilNormY)
	# ax2.plot(pupilInterpolatedTimeStamp[Ypeakmax], y_list[Ypeakmax], 'rx', markersize=10)
	# ax2.plot(pupilInterpolatedTimeStamp[Ypeakmin], y_list[Ypeakmin], 'kx', markersize=10)

	Xpeakmax += Xpeakmin
	Xpeakmax += Ypeakmax
	Xpeakmax += Ypeakmin
	finalPeaks = list(set(Xpeakmax))
	finalPeaks.sort()

	Xpeakpoints = [x_list[0]]
	Xpeakpoints += x_list[finalPeaks]
	Xpeakpoints += [x_list[-1]]

	Ypeakpoints = [y_list[0]]
	Ypeakpoints += y_list[finalPeaks]
	Ypeakpoints += [y_list[-1]]

	Xvector = np.diff(Xpeakpoints)
	Yvector = np.diff(Ypeakpoints)
	vectors = np.column_stack((Xvector, Yvector))

	plt.figure(figsize=(7, 7))
	plt.scatter(vectors[:, 0], vectors[:, 1], alpha=0.8)
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')

	pca = PCA(n_components=2)
	pca.fit(vectors)

	first_axis = pca.components_[0]

	# print(pca.explained_variance_)
	# print(pca.components_)
	# print(pca.explained_variance_ratio_)
	for length, vector in zip(pca.explained_variance_, pca.components_):
		v = vector * 3 * np.sqrt(length)
		# v = vector * 3 * length
		draw_vector(pca.mean_, pca.mean_ + v)
	plt.axis('equal')

	''' #dimension reduction
	pca1 = PCA(n_components=1)
	pca1.fit(vectors)
	X_pca = pca1.transform(vectors)
	X_new = pca1.inverse_transform(X_pca)
	plt.scatter(X_new[:, 0], X_new[:, 1])
	'''


# convert euler angle into ~0 degree to make calculation easier
def convert_angle(angle):
	if angle > 180:
		return angle - 360
	else:
		return angle


def draw_vector(v0, v1, ax=None):
	ax = ax or plt.gca()
	arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
	ax.annotate('', v1, v0, arrowprops=arrowprops)
