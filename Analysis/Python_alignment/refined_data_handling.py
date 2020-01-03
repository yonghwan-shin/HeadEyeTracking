from fileHandling import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.signal
from peakdetect import *
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
	plt.rcParams["figure.figsize"]= (12,6)
	plt.rcParams["axes.grid"] = True
	imu = imu_data.query("ImuTimeStamp >= 1950 and ImuTimeStamp <=6500")
	pupil = pupil_data.query("timestamp >= 1950 and timestamp <=6500")
	# imu = imu_data
	# pupil = pupil_data
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
	# timestamp = np.linspace(0,6000,6001)
	xhat = scipy.signal.savgol_filter(interpolate_normalized_imu_X(timestamp), 255, 2)
	yhat = scipy.signal.savgol_filter(interpolate_normalized_pupil_Y(timestamp),255,2)
	xhat = normalize(xhat)
	yhat = normalize(yhat)

	try:
		Xpeaks = peakdetect(xhat, lookahead=15)
		Xpeaksmax = Xpeaks[0]
		Xpeaksmin = Xpeaks[1]
		Xpeakmax = [Xpeaksmax[i][0] for i in range(len(Xpeaksmax))]
		Xpeakmin = [Xpeaksmin[i][0] for i in range(len(Xpeaksmin))]
		Ypeaks = peakdetect(yhat, lookahead=15)
		Ypeaksmax = Ypeaks[0]
		Ypeaksmin = Ypeaks[1]
		Ypeakmax = [Ypeaksmax[i][0] for i in range(len(Ypeaksmax))]
		Ypeakmin = [Ypeaksmin[i][0] for i in range(len(Ypeaksmin))]
		Xpeakmax += Xpeakmin
		# Xpeakmax += Ypeakmax
		# Xpeakmax += Ypeakmin
		Ypeakmax += Ypeakmin
		XfinalPeaks = list(set(Xpeakmax))
		YfinalPeaks = list(set(Ypeakmax))
		XfinalPeaks.sort()
		YfinalPeaks.sort()
		Xdiff = np.diff(xhat[XfinalPeaks])
		Ydiff = np.diff(yhat[YfinalPeaks])
		X_positive = Xdiff[Xdiff>=0]
		X_negative = Xdiff[Xdiff<0]
		Y_positive = Ydiff[Ydiff>=0]
		Y_negative = Ydiff[Ydiff<0]
		avg_X_positive = np.average(X_positive)
		avg_X_negative = np.average(X_negative)
		avg_Y_positive = np.average(Y_positive)
		avg_Y_negative = np.average(Y_negative)
		# print(avg_X_positive,avg_X_negative,avg_Y_positive,avg_Y_negative)
		# xhat *= (avg_Y_positive-avg_Y_negative) / (avg_X_positive-avg_X_negative)
		# yhat *= (1/2)*(avg_X_positive-avg_X_negative) / (avg_Y_positive-avg_Y_negative)
		plt.plot(xhat,'r',linestyle ='dashed' ,label="imu X")
		plt.plot(yhat,'b', linestyle = 'dashed',label = "pupil Y")
		plt.plot((xhat+yhat)/2,'k',label = "Compensation")
		plt.legend(loc= "uppemr left")
	except:
		print("error in timestamp")

	plt.suptitle(trial_info)
	plt.show()
	pass

def normalize(df):
	max = np.max(df)
	min = np.min(df)
	output = (df - min)/(max-min)
	# print(max-min)
	return output