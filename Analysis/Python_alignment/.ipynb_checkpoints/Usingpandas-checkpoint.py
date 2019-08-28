from fileHandling import *


rootDirectory = '/Users/yonghwanshin/OneDrive - unist.ac.kr/Research/2019_VOR_VR/Datasets/1stData/'
processingDirectory = 'Processing_'
hololensDirectory = 'result_sub'




##############################
# Main function
##############################

for subjectNum in range(1, 2):
    # get file names
    processingFileNameList = getFileNameList(
        rootDirectory, processingDirectory, subjectNum)
    hololensFileNameList = getFileNameList(
        rootDirectory, hololensDirectory, subjectNum)
    # processingFileNameList.sort()
    # hololensFileNameList.sort()

    # for fileCount in range(1,len(processingFileNameList)):
    for fileCount in range(1, 2):

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

        if HololensData == []: break

        plt.title('test')

        # print(ProcessingData[0])
        imuTime = ProcessingData[1][2].astype(float)
        imuQuaternionX = ProcessingData[1][3].astype(float)
        imuQuaternionY = ProcessingData[1][4].astype(float)
        imuQuaternionZ = ProcessingData[1][5].astype(float)
        imuQuaternionW = ProcessingData[1][6].astype(float)

        # specify the data type...
        # timestamp (time since app starts)
        x = HololensData[0][3].astype(float)
        # hololens z vector
        y = HololensData[0][8].astype(float)

        
        peaks, _ = find_peaks(y, height=0)
        plt.plot( y)
        plt.plot(peaks, y[peaks], "x")

        # plt.plot(x, y)
        # plt.show()
        # plt.title('imu')
        # plt.plot(imuQuaternionX)
        # plt.plot(imuQuaternionY)
        # plt.plot(imuQuaternionZ)
        # plt.plot(imuQuaternionW)
        plt.show()
