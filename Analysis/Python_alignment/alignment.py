import csv
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as numpy
import scipy as scipy
from scipy.signal import find_peaks_cwt
import fileHandling


def getFileNameList(rootDirectory, targetDirectory, subjectNumer):
    fileList = []
    fullPath = rootDirectory + targetDirectory + str(getSubject(subjectNumer))
    for (path, dir, files) in os.walk(fullPath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                fileList.append(filename)
    return fileList


def searchFiles(dirName):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print(full_filename)


def checkProcessingFile(fileDirectory):
    with open(fileDirectory, 'r') as f:
        d = pd.read_csv(f)

        reader = csv.reader(f)
        checkFileName = next(reader)
        headers = next(reader)
        data = list(reader)
        data = numpy.array(data)

    pupildataLength = headers[3]
    pupildataLength = int(pupildataLength)
    imudataLength = headers[5]
    imudataLength = int(imudataLength)
    pupilData = data[:pupildataLength]
    imuData = data[pupildataLength:]
 # check correct file...
    if len(pupilData) == pupildataLength and len(imuData) == imudataLength:
        # print('processing file is looking fine...')
        return [pupilData, imuData]


def checkHololensFile(fileDirectory):
    with open(fileDirectory, 'r') as f:
        reader = csv.reader(f)
        checkFileName = next(reader)
        headers = next(reader)
        data = list(reader)
        data = numpy.array(data)
    for i in range(1, len(data)):
        if(data[i][0] == 'SUMMARY:'):
            sep = i
            break
    mainData = data[0:sep]
    summaryData = data[sep:]
    return [mainData, summaryData]


def getSubject(num):
    if num is not 9:
        return num
    else:
        return 109


# Main function
rootDirectory = '/Users/yonghwanshin/OneDrive - unist.ac.kr/Research/2019_VOR_VR/Datasets/1stData/'
processingDirectory = 'Processing_'
hololensDirectory = 'result_sub'


for subjectNum in range(1, 2):
        # processing files
    processingFileNameList = getFileNameList(
        rootDirectory, processingDirectory, subjectNum)
    hololensFileNameList = getFileNameList(
        rootDirectory, hololensDirectory, subjectNum)
#     for fileCount in range(1,len(processingFileNameList)):
for fileCount in range(1, 2):
    ProcessingData = checkProcessingFile(rootDirectory + processingDirectory +
                                         str(getSubject(subjectNum))+"/"+processingFileNameList[fileCount])
    holoFileNum = hololensFileNameList.index(processingFileNameList[fileCount])

    HololensData = checkHololensFile(rootDirectory + hololensDirectory +
                                     str(getSubject(subjectNum))+"/"+hololensFileNameList[holoFileNum])
#     indexes = find_peaks_cwt(ProcessingData[1],numpy.arange(1,500))


# for subject in range(1,17):
#     subjectNumber = getSubject(subject)
#     processingFile = rootDirectory + processingDirectory + str(subjectNumber) + fileName
#     hololensFile = rootDirectory + hololensDirectory + str(subjectNumber) + fileName
