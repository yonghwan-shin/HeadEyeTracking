import csv
import os
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
# import scipy.signal
from scipy.signal import find_peaks


def makeTrialInfo(info):
    target = info[0]
    env = info[1]
    pos = info[2]
    block = info[3]
    c = info[4]
    output = 'T' + str(target) + "_E" + str(env) + '_P' + str(pos) + '_B' + str(block) + '_C' + str(c)
    return output


def getTrialInfo(fileName):
    output = [0, 0, 0, 0, 0]
    try:
        target = fileName[1]
        env = fileName[4]
        pos = fileName[7]
        block = fileName[10]
        c = fileName[13]
        output = [target, env, pos, block, c]
    except:
        print('something wrong in filename...')
    return output


def getFileNameList(rootDirectory, targetDirectory, subjectNumer):
    fileList = []
    fullPath = rootDirectory + targetDirectory + str(getSubject(subjectNumer))
    for (path, dir, files) in os.walk(fullPath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                fileList.append(filename)
    return fileList


def getSubject(num):
    if num is not 9:
        return num
    else:
        return 109


def searchFiles(dirName):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print(full_filename)


def getProcessingFile(filename):
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
        columns2 = ['UDPTimeStamp','PupilTimeStamp','ImuTimeStamp','QuatI','QuatJ','QuatK','QuatReal','QuatRadianAccuracy','pIntersectX','pIntersectZ']
        data1 = data1.append(data.loc[:int(pupilLength) - 1], ignore_index=True)
        data2 = data2.append(data.loc[int(pupilLength):], ignore_index=True)
        data1 = data1.astype(float, errors='ignore')
        data2 = data2.astype(float, errors='ignore')
        data2.drop(data2.columns[range(10,41)],axis=1,inplace=True)
        data2.columns = columns2
    return [data1, data2]


def getHololensFile(filename):
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

    return [data1, data2]


