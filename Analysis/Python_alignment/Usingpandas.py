from fileHandling import *


from subjectHandling import *

for subjectNum in range(1,2):
    # for subjectNum in range(1, 2):
    fileLists = getOneSubject(subjectNum)

    # for fileCount in range(0,len(fileLists[0])):
    for fileCount in range(0,1):
        Files = getOneFile(fileLists[0],fileLists[1],subjectNum,fileCount)
        analyzeFile(Files[0],Files[1])

    print('-'*80)
##############################
# Main function
##############################
#
# for subjectNum in range(1, 2):
#     # get file names
#     processingFileNameList = getFileNameList(
#         rootDirectory, processingDirectory, subjectNum)
#     hololensFileNameList = getFileNameList(
#         rootDirectory, hololensDirectory, subjectNum)
#     # processingFileNameList.sort()
#     # hololensFileNameList.sort()
#
#     # for fileCount in range(1,len(processingFileNameList)):
#     for fileCount in range(1, 2):
#
#         print("Analyzing...  " + processingFileNameList[fileCount])
#         ProcessingData = getProcessingFile(rootDirectory + processingDirectory +
#                                            str(getSubject(subjectNum)) + "/" + processingFileNameList[fileCount])
#         HololensData = []
#         # find holo data
#         for name in hololensFileNameList:
#             trialDetail = getTrialInfo(processingFileNameList[fileCount])
#             if name[:14] == makeTrialInfo(trialDetail):
#                 HololensData = getHololensFile(rootDirectory + hololensDirectory +
#                                                str(getSubject(subjectNum)) + "/" + name)
#                 break
#             else:
#                 pass
#
#         if HololensData == []: break
#
#         imuTimeStamp = ProcessingData[1]['ImuTimeStamp'].astype(float)
#         pIntersectZ = ProcessingData[1]['pIntersectZ'].astype(float)
#         QuatI = ProcessingData[1]['QuatI']
#         QuatJ = ProcessingData[1]['QuatJ']
#         QuatK = ProcessingData[1]['QuatK']
#         QuatReal = ProcessingData[1]['QuatReal']
#
#
#         HoloTimeStamp = HololensData[0]['UTC'].astype(float)
#         HeadForwardVectorZ = HololensData[0]['HeadForwardVectorZ'].astype(float)
#
#         peaks, _ = find_peaks(HeadForwardVectorZ)
#         # peaks, _ = find_peaks(HeadForwardVectorZ, height=0)
#         peaks2, _ = find_peaks(pIntersectZ)
#         fig = plt.figure()
#         ax1 = fig.add_subplot(2, 1, 1)
#         ax2 = fig.add_subplot(2, 1, 2)
#         f = signal.resample(HeadForwardVectorZ,1169)
#
#         ax1.plot(f)
#         ax2.plot(pIntersectZ)
#
#
#         # # ax1.title('head vector Z')
#         # ax1.plot(HoloTimeStamp, HeadForwardVectorZ)
#         # ax1.plot(HoloTimeStamp[peaks], HeadForwardVectorZ[peaks], "rx")
#         #
#         # # ax2.title('intersect Z')
#         # ax2.plot(imuTimeStamp, pIntersectZ)
#         # # ax2.plot(imuTimeStamp[peaks2],pIntersectZ[peaks2],'rx')
#         #
#         # s = chisqr_align(HeadForwardVectorZ,pIntersectZ)
#         # ax2.plot(shift(imuTimeStamp,s,mode='nearest'),pIntersectZ, ls='--')
#         # ax1.plot(HeadForwardVectorZ)
#         # ax2.plot(pIntersectZ)
#         # print(len(HeadForwardVectorZ))
#         # print(len(pIntersectZ))
#         # s = chisqr_align(f, pIntersectZ)
#         s = phase_align(f,pIntersectZ,[10,500])
#         ax2.plot(shift(pIntersectZ,s,mode='nearest'),ls='--',label='aligned data')
#         print('phase shift value to align is ', s)
#
#
#         r = R.from_quat([QuatReal[0],QuatI[0],QuatJ[0],QuatK[0]])
#         print(r.as_euler('zxy',degrees=True))
#
#         plt.show()
