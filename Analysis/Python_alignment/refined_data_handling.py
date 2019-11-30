from fileHandling import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.signal
def get_refined_files(filename):
	currentDirectory = os.getcwd()
	file_path = os.path.join(currentDirectory, "refined_data_saved", filename)
	if not os.path.isfile(file_path):
		raise ValueError("there is no file:" + filename)
	file = pd.read_csv(file_path)
	return file

# [ pupil data ]
# timestamp
# norm_posX
# norm_posY
# confidence
# theta
# phi

# [ imu data ]
# UDPTimeStamp
# PupilTimeStamp
# * ImuTimeStamp
# QuatI
# QuatJ
# QuatK
# QuatReal
# QuatRadianAccuracy
# pIntersectX
# pIntersectZ
# * angleX
# * angleY
# * angleZ

def look_refined_data(imu_data, pupil_data,trial_info):
	# graph settings
	plt.rcParams["figure.figsize"]= (6,3)
	plt.rcParams["axes.grid"] = True
	imu = imu_data.query("ImuTimeStamp >= 1950 and ImuTimeStamp <=6500")
	pupil = pupil_data.query("timestamp >= 1950 and timestamp <=6500")
	# normalized_imu_X = normalize(imu["angleX"])
	# normalized_pupil_Y = normalize(pupil["norm_posY"])

	interpolate_normalized_imu_X = interpolate.interp1d(imu["ImuTimeStamp"],imu["angleX"])
	interpolate_normalized_pupil_Y = interpolate.interp1d(pupil["timestamp"],pupil["norm_posY"])
	start1 = imu["ImuTimeStamp"].head(1).values[0]
	start2 = pupil["timestamp"].head(1).values[0]
	end1 = imu["ImuTimeStamp"].tail(1).values[0]
	end2 = pupil["timestamp"].tail(1).values[0]
	if start2 >2000:
		raise ValueError("Too short pupil data", trial_info)
	timestamp = np.linspace(2000,6000,4001)
	xhat = scipy.signal.savgol_filter(interpolate_normalized_imu_X(timestamp), 255, 2)
	yhat = scipy.signal.savgol_filter(interpolate_normalized_pupil_Y(timestamp),255,2)
	xhat = normalize(xhat)
	yhat = normalize(yhat)
	# print(interpolate_normalized_imu_X(timestamp))
	# print(interpolate_normalized_imu_X(timestamp) + interpolate_normalized_pupil_Y(timestamp))
	try:
		# plt.plot(timestamp,interpolate_normalized_imu_X(timestamp),'r')
		# plt.plot(timestamp,interpolate_normalized_pupil_Y(timestamp),'b')
		# add = interpolate_normalized_imu_X(timestamp) - interpolate_normalized_pupil_Y(timestamp)
		# plt.plot(interpolate_normalized_imu_X(timestamp),'b')
		# plt.plot(interpolate_normalized_pupil_Y(timestamp),'b')
		plt.plot(xhat,'r')
		# plt.plot(yhat,'b')
		# plt.plot((xhat+yhat)/2,'x')
		# plt.plot(xhat-yhat)
		# plt.plot(add)
		# plt.plot(timestamp)
	except:
		print("error in timestamp")

	plt.suptitle(trial_info)
	plt.show()
	pass

def normalize(df):
	max = np.max(df)
	min = np.min(df)
	output = (df - min)/(max-min)
	print(max-min)
	return output