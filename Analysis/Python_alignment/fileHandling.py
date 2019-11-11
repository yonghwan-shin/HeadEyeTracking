import csv
import os
import time

import demjson
import pandas as pd
from quaternionTest import *

# import scipy.signal

ROOT_DIRECTORY = '/Users/yonghwanshin/Documents/GitHub/HeadEyeTracking/Datasets/1stData/'
PROCESSING_DIRECTORY = 'Processing_'
HOLOLENS_DIRECTORY = 'result_sub'


def make_trial_info(info):
	target = info[0]
	env = info[1]
	pos = info[2]
	block = info[3]
	# c = info[4]
	# output = 'T' + str(target) + "_E" + str(env) + '_P' + str(pos) + '_B' + str(block) + '_C' + str(c)
	output = 'T' + str(target) + "_E" + str(env) + '_P' + str(pos) + '_B' + str(block)
	return output


def get_trial_info(fileName):
	output = [0, 0, 0, 0]
	try:
		target = fileName[1]
		env = fileName[4]
		pos = fileName[7]
		block = fileName[10]
		# c = fileName[13]
		output = [target, env, pos, block]
	except:
		print('something wrong in filename...')
	return output


def get_filename_list(root_directory, target_directory, subject_number):
	fileList = []
	fullPath = root_directory + target_directory + str(get_subject(subject_number))
	for (path, dir, files) in os.walk(fullPath):
		for filename in files:
			ext = os.path.splitext(filename)[-1]
			if ext == '.csv':
				fileList.append(filename)
	# fileList.sort()
	return fileList


def get_subject(num):
	if num is not 9:
		return num
	else:
		return 109


def search_files(directory_name):
	filenames = os.listdir(directory_name)
	for filename in filenames:
		full_filename = os.path.join(directory_name, filename)
		print(full_filename)


def get_processing_file(filename):
	prev = time.time()
	reader = []
	if not os.path.isfile(filename):
		csvfile = open(filename, "w")
	else:
		csvfile = open(filename, "r")
		reader = csv.reader(csvfile, delimiter=',')
		a = next(reader)
		check = next(reader)
		pupilLength = check[3]
		imuLength = check[5]
		data = pd.DataFrame(reader)

		data1 = pd.DataFrame()
		data2 = pd.DataFrame()
		columns2 = ['UDPTimeStamp', 'PupilTimeStamp', 'ImuTimeStamp', 'QuatI', 'QuatJ', 'QuatK', 'QuatReal',
		            'QuatRadianAccuracy', 'pIntersectX', 'pIntersectZ']
		data1 = data1.append(data.loc[:int(pupilLength) - 1], ignore_index=True)
		data2 = data2.append(data.loc[int(pupilLength):], ignore_index=True)
		data1 = data1.astype(float, errors='ignore')

		pupil = data1[data1.columns[6:]]
		pupilData = []
		pupilInitTime = 0;
		for index, row in pupil.iterrows():
			a = row.values
			s = ""
			for i in a:
				s = s + "," + str(i)
			s = s[1:]
			json_dict = demjson.decode(s)
			# pupil timestamp is re-shaped(0 to ~6500)
			if pupilInitTime == 0:
				pupilInitTime = json_dict['timestamp']
			json_dict['timestamp'] = float(json_dict['timestamp']) - pupilInitTime
			pupilData.append(json_dict)

		data2 = data2.astype(float, errors='ignore')
		data2.drop(data2.columns[range(10, 41)], axis=1, inplace=True)
		data2.columns = columns2
		data2 = data2.drop_duplicates(subset='ImuTimeStamp', keep='first')
		if data2['ImuTimeStamp'].size>100:
			data2['ImuTimeStamp'] = data2['ImuTimeStamp'] - data2['ImuTimeStamp'].head(1).values[0]
		else:
			return[None,None]
	# IMU Timestamp is re-shaped(0 to ~6500)
	# print(time.time() - prev, ' second used to prepare processing file')
	return [pupilData, data2]


def get_hololens_file(filename):
	startTime = time.time()
	reader = []
	if not os.path.isfile(filename):
		csvfile = open(filename, "w")
	else:
		csvfile = open(filename, "r")
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		header = next(reader)
		data = pd.DataFrame(reader)
		count = 0
		for i in range(0, len(data[0])):
			if data[0][i] == 'SUMMARY:':
				count = i
		data1 = pd.DataFrame()
		data2 = pd.DataFrame()
		data1 = data1.append(data.loc[:count - 1], ignore_index=True)
		HoloColumn = ['SaveFileName', 'FrameCount', 'UTC', 'TimeSinceStart', 'HeadPositionX', 'HeadPositionY',
		              'HeadPositionZ', 'HeadForwardVectorX', 'HeadForwardVectorY', 'HeadForwardVectorZ', 'HeadAngleX',
		              'HeadAngleY', 'HeadAngleZ', 'HeadQuaternionX', 'HeadQuaternionY', 'HeadQuaternionZ',
		              'HeadQuaternionW', 'TargetPositionX', 'TargetPositionY', 'TargetPositionZ', 'AngleDifference',
		              'OverTarget']
		data1.columns = HoloColumn
		data2 = data2.append(data.loc[count:], ignore_index=True)
		data1 = data1.astype(float, errors='ignore')
		data1['UTC'] = data1['UTC'].astype(float, errors='ignore')
		data2 = data2.astype(float, errors='ignore')
		for i in range(0, len(data1['HeadAngleX'])):
			if float(data1['HeadAngleX'][i]) > 180.0:
				data1.at[i, 'HeadAngleX'] = float(data1['HeadAngleX'][i]) - 360
			if float(data1['HeadAngleY'][i]) > 180.0:
				data1.at[i, 'HeadAngleY'] = float(data1['HeadAngleY'][i]) - 360
			if float(data1['HeadAngleZ'][i]) > 180.0:
				data1.at[i, 'HeadAngleZ'] = float(data1['HeadAngleZ'][i]) - 360
		# data1.drop_duplicates()
		# data2.drop_duplicates()
		data1['UTC'] = data1['UTC'] - data1['UTC'].head(1).values[0]
	# Hololens UTC timestamp changed... (0 to ~6500)
	# print((time.time()-startTime), ' second used to prepare Hololens file')
	return [data1, data2]


# get eye values from pupil-cam
def organise_pupil_data(pupilDataFile):

	pupilData = pd.DataFrame(columns=['timestamp', 'norm_posX', 'norm_posY', 'confidence','theta','phi'])
	if pupilDataFile is None:
		return [None, None, None, None, None, None]
	pupilTimeStamp = []
	pupilNorm_PosX = []
	pupilNorm_PosY = []
	pupilConfidence = []
	pupilTheta = []
	pupilPhi = []
	for i in pupilDataFile:
		pupilTimeStamp.append(i['timestamp'])
		pupilNorm_PosX.append(i['norm_pos'][0])
		pupilNorm_PosY.append(i['norm_pos'][1])
		pupilConfidence.append(i['confidence'])
		pupilTheta.append(i['theta'])
		pupilPhi.append(i['phi'])
	pupilData['timestamp'] = pupilTimeStamp
	pupilData['norm_posX'] = pupilNorm_PosX
	pupilData['norm_posY'] = pupilNorm_PosY
	pupilData['confidence'] = pupilConfidence
	pupilData['theta'] = pupilTheta
	pupilData['phi'] = pupilPhi
	# TODO: drop low confidnece values...? -> set threshold as .6
	confidenceThreshold = 0.6
	pupilData = pupilData[pupilData.confidence > confidenceThreshold]
	pupilData.drop_duplicates('timestamp')
	pupilTimeStamp = pupilData['timestamp']
	pupilNorm_PosX = pupilData['norm_posX']
	pupilNorm_PosY = pupilData['norm_posY']
	pupilConfidence = pupilData['confidence']
	pupilTheta = pupilData['theta']
	pupilPhi = pupilData['phi']

	return [pupilTimeStamp, pupilNorm_PosX, pupilNorm_PosY, pupilConfidence,pupilTheta,pupilPhi]


def organise_imu_data(imu_data):
	imuTimeStamp = imu_data['ImuTimeStamp'].astype(float)
	quaternions = imu_data[['QuatReal', 'QuatI', 'QuatJ', 'QuatK']]
	angleX = []
	angleY = []
	angleZ = []
	for index, row in quaternions.iterrows():
		r = R.from_quat([row.QuatI, row.QuatJ, row.QuatK, row.QuatReal])
		quaternion = quaternion_to_euler(row.QuatI, row.QuatJ, row.QuatK, row.QuatReal)
		angleX.append(-quaternion[0])
		angleY.append(-quaternion[1])
		angleZ.append(-90 - quaternion[2])
	imu_data['angleX'] = angleX
	imu_data['angleY'] = angleY
	imu_data['angleZ'] = angleZ
	AngleX = imu_data['angleX'].astype(float)
	AngleY = imu_data['angleY'].astype(float)
	AngleZ = imu_data['angleZ'].astype(float)
	return [AngleX, AngleY, AngleZ, imuTimeStamp]
